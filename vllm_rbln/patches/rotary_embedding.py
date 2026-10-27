# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import rotate_gptj, rotate_neox
from vllm.model_executor.models.utils import extract_layer_index

from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch

logger = init_logger(__name__)

original_rope_init = RotaryEmbedding.__init__


@register_patch(
    target="vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.__init__",
    reason=(
        "The RBLN path needs rotate-half style RoPE caches because that "
        "cache layout is more compatible with RBLN execution than the "
        "upstream layout."
    ),
)
def patched_rope_init(
    self: RotaryEmbedding,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
    init_cache: bool = True,
):
    original_rope_init(
        self,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        init_cache,
    )

    # For best compatibility with rbln, we use the rotate_half-style RoPE.
    cos, sin = self.cos_sin_cache.chunk(2, dim=-1)
    if self.is_neox_style:
        cos = cos.repeat(1, 2)
        sin = sin.repeat(1, 2)
    else:
        cos = torch.stack([cos, cos], dim=-1).reshape(cos.shape[0], -1)
        sin = torch.stack([sin, sin], dim=-1).reshape(sin.shape[0], -1)
    self.register_buffer("cos_cache", cos, persistent=False)
    self.register_buffer("sin_cache", sin, persistent=False)


@register_patch(
    target="vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.forward_oot",
    reason=(
        "The RBLN path needs a dedicated RoPE execution path that consumes "
        "the rotate-half cache layout prepared at initialization and "
        "applies rotary embeddings with RBLN-friendly tensor layouts."
    ),
)
def patched_rope_forward_oot(
    self: RotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: torch.Tensor | None = None,
    *,
    q_hadamard: torch.Tensor | None = None,
    k_hadamard: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        positions: [batch_size, seq_len]
        query: [batch_size, seq_len, num_heads * head_size]
        key: [batch_size, seq_len, num_kv_heads * head_size]
        q_hadamard, k_hadamard: [rotary_dim, rotary_dim], applied to every
            head after RoPE. Only `QKHadamardRotaryEmbedding` passes them.
    """
    # NOTE(RBLN): For best compatibility with rbln,
    # tensors are reshaped/transposed as follows:
    # - cos, sin: (batch_size, 1, seq_len, rotary_dim * 2)
    # - query, key: (batch_size, num_heads, seq_len, head_size)

    if offsets is not None:
        positions = positions + offsets

    batch_size, seq_len = positions.shape[0], positions.shape[1]
    rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj

    positions_flat = positions.flatten()
    cos = (
        self.cos_cache.index_select(0, positions_flat)
        .view(batch_size, 1, seq_len, -1)
        .to(query.dtype)
    )
    sin = (
        self.sin_cache.index_select(0, positions_flat)
        .view(batch_size, 1, seq_len, -1)
        .to(query.dtype)
    )

    query_shape = query.shape
    query = query.view(batch_size, seq_len, -1, self.head_size)
    query_rot = query[..., : self.rotary_dim]
    query_rot = query_rot.transpose(1, 2)
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    # The rotation sits in the [batch, heads, seq, dim] layout, right after
    # RoPE's add. Placed after the transpose back, the decode graph fails to
    # compile (OpInvalidWeightSharingError).
    if q_hadamard is not None:
        query_rot = torch.matmul(query_rot, q_hadamard)
    query_rot = query_rot.transpose(1, 2)
    # FIXME(RBLN) - if slice size is zero, DO NOT slice
    if self.head_size == self.rotary_dim:
        query = query_rot.reshape(query_shape)
    else:
        query_pass = query[..., self.rotary_dim :]
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(batch_size, seq_len, -1, self.head_size)
    key_rot = key[..., : self.rotary_dim]
    key_rot = key_rot.transpose(1, 2)
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin
    if k_hadamard is not None:
        key_rot = torch.matmul(key_rot, k_hadamard)
    key_rot = key_rot.transpose(1, 2)
    # FIXME(RBLN) - if slice size is zero, DO NOT slice
    if self.head_size == self.rotary_dim:
        key = key_rot.reshape(key_shape)
    else:
        key_pass = key[..., self.rotary_dim :]
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    return query, key


class QKHadamardRotaryEmbedding(torch.nn.Module):
    """One attention layer's RoPE, followed by a Hadamard rotation of every
    query and key head.

    With H the Sylvester Hadamard matrix of order d = 2^n, H H^T = d I, so
    (q H / 2^floor(n/2)) (k H / 2^ceil(n/2))^T = q k^T. Both scales are powers
    of two, which keeps the scaled matrices exact in fp32, fp16 and bf16. The KV
    cache stores the rotated keys: RoPE's large low-frequency key channels are
    spread over the head before the cache quantizes them.
    """

    def __init__(
        self, rope: RotaryEmbedding, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        dim = rope.rotary_dim
        if dim & (dim - 1):
            raise ValueError(
                f"use_qk_hadamard needs a power-of-two rotary_dim, got {dim}"
            )
        hadamard = torch.ones(1, 1)
        while hadamard.shape[0] < dim:
            hadamard = torch.cat(
                [
                    torch.cat([hadamard, hadamard], 1),
                    torch.cat([hadamard, -hadamard], 1),
                ]
            )
        n = dim.bit_length() - 1
        self.rope = rope
        self.q_hadamard = torch.nn.Parameter(
            (hadamard * 2.0 ** -(n // 2)).to(device=device, dtype=dtype),
            requires_grad=False,
        )
        self.k_hadamard = torch.nn.Parameter(
            (hadamard * 2.0 ** (n // 2 - n)).to(device=device, dtype=dtype),
            requires_grad=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return patched_rope_forward_oot(
            self.rope,
            positions,
            query,
            key,
            offsets,
            q_hadamard=self.q_hadamard,
            k_hadamard=self.k_hadamard,
        )


def rotate_qk_heads(
    model: torch.nn.Module, skip_layers: list[int], dtype: torch.dtype
) -> None:
    """Give every attention layer that has a `rotary_emb`, except those in
    `skip_layers`, its own `QKHadamardRotaryEmbedding`.

    `get_rope` hands every layer the same RoPE module, so the rotation cannot
    be a flag on that module if some layers are to stay unrotated.
    """
    attention = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(getattr(module, "rotary_emb", None), RotaryEmbedding)
    ]
    layers = set()
    for name, module in attention:
        rope = module.rotary_emb
        if type(rope).forward_oot is not patched_rope_forward_oot:
            raise ValueError(
                f"use_qk_hadamard covers the base RoPE only; {name} uses "
                f"{type(rope).__name__}, which has a forward of its own"
            )
        layer = extract_layer_index(name)
        layers.add(layer)
        if layer in skip_layers:
            continue
        device = next(module.parameters()).device
        module.rotary_emb = QKHadamardRotaryEmbedding(rope, dtype, device)
    if not layers:
        raise ValueError("use_qk_hadamard is set, but no layer has a rotary_emb")
    if unknown := sorted(set(skip_layers) - layers):
        raise ValueError(
            f"qk_hadamard_skip_layers names {unknown}, which have no rotary_emb; "
            f"the layers with one are {sorted(layers)}"
        )
    logger.info(
        "Rotating query and key heads after RoPE in %d of %d attention layers "
        "(skipped: %s).",
        len(layers - set(skip_layers)),
        len(layers),
        sorted(skip_layers),
    )
