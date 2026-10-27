# Copyright 2026 Rebellions Inc. All rights reserved.
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

from typing import Any

import pytest
import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from vllm_rbln.patches.rotary_embedding import (
    QKHadamardRotaryEmbedding,
    rotate_qk_heads,
)

_HEAD_SIZE = 128


def _rope(rotary_dim=_HEAD_SIZE, is_neox_style=True) -> RotaryEmbedding:
    # CustomOp.__init__ needs a built VllmConfig; the patched forward_oot reads
    # only these attributes.
    rope: Any = object.__new__(RotaryEmbedding)
    torch.nn.Module.__init__(rope)
    rope.head_size = _HEAD_SIZE
    rope.rotary_dim = rotary_dim
    rope.is_neox_style = is_neox_style
    angles = torch.outer(torch.arange(32.0), torch.linspace(0.01, 1.0, rotary_dim))
    rope.cos_cache = angles.cos()
    rope.sin_cache = angles.sin()
    return rope


def _scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Every query head against every key head: [batch, q head, k head, s, t]."""
    b, s, _ = q.shape
    q = q.view(b, s, -1, _HEAD_SIZE)
    k = k.view(b, s, -1, _HEAD_SIZE)
    return torch.einsum("bshd,btgd->bhgst", q, k)


@pytest.mark.parametrize("rotary_dim", [_HEAD_SIZE, _HEAD_SIZE // 2])
@pytest.mark.parametrize("is_neox_style", [True, False])
def test_rotation_keeps_the_scores_and_changes_the_keys(rotary_dim, is_neox_style):
    rope = _rope(rotary_dim, is_neox_style)
    torch.manual_seed(0)
    positions = torch.arange(10).view(2, 5)
    q = torch.randn(2, 5, 4 * _HEAD_SIZE)
    k = torch.randn(2, 5, 2 * _HEAD_SIZE)

    q_ref, k_ref = rope.forward_oot(positions, q.clone(), k.clone())
    rotated = QKHadamardRotaryEmbedding(rope, torch.float32, torch.device("cpu"))
    q_rot, k_rot = rotated(positions, q.clone(), k.clone())

    torch.testing.assert_close(
        _scores(q_rot, k_rot), _scores(q_ref, k_ref), atol=1e-4, rtol=1e-4
    )
    assert not torch.allclose(k_rot, k_ref, atol=1e-2)


class _Layer(torch.nn.Module):
    def __init__(self, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.qkv_proj = torch.nn.Linear(4, 4)
        self.rotary_emb = rope


def _model(num_layers: int) -> torch.nn.Module:
    model = torch.nn.Module()
    rope = _rope()
    model.layers = torch.nn.ModuleList(_Layer(rope) for _ in range(num_layers))
    return model


def test_skipped_layers_keep_the_shared_rope():
    model = _model(3)
    rope = model.layers[0].rotary_emb

    rotate_qk_heads(model, [0], torch.float32)

    assert model.layers[0].rotary_emb is rope
    for layer in model.layers[1:]:
        assert isinstance(layer.rotary_emb, QKHadamardRotaryEmbedding)
        assert layer.rotary_emb.rope is rope


def test_a_skip_layer_without_rope_is_rejected():
    with pytest.raises(ValueError, match=r"names \[5\]"):
        rotate_qk_heads(_model(3), [5], torch.float32)
