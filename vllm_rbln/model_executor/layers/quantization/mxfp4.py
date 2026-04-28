# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import vllm_rbln.rbln_envs as envs


def _dequantize_mxfp4(
    blocks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """
    Args:
        blocks: uint8 [..., K // 2] containing packed FP4 values
        scales: uint8 [..., K // 32] containing E8M0 scales
        dtype: output dtype

    Returns:
        Dequantized tensor of shape [..., K]
    """
    # fmt: off
    FP4_VALUES = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    # fmt: on
    lut = torch.tensor(FP4_VALUES, dtype=dtype)

    # Convert E8M0 scales to exponents (subtract bias of 127)
    exponents = scales.to(torch.int32) - 127

    # Unpack FP4 nibbles
    idx_lo = (blocks & 0x0F).to(torch.long)
    idx_hi = (blocks >> 4).to(torch.long)

    # Look up FP4 values
    val_lo = lut[idx_lo]
    val_hi = lut[idx_hi]

    # Interleave low and high nibbles
    *prefix_shape, B = blocks.shape
    out = torch.empty(*prefix_shape, B * 2, dtype=dtype)
    out[..., 0::2] = val_lo
    out[..., 1::2] = val_hi

    # Apply scales: each scale covers 32 elements
    # scales shape: [..., K // 32], out shape: [..., K]
    # Expand scales to match output shape
    exponents_expanded = exponents.unsqueeze(-1).expand(*exponents.shape, 32)
    exponents_expanded = exponents_expanded.reshape(*prefix_shape, -1)

    # ldexp: out * 2^exponents
    out = torch.ldexp(out, exponents_expanded[..., : out.shape[-1]])

    return out


def _swigluoai(
    gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float
) -> torch.Tensor:
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


# kernel for gpt_oss, with built-in swigluoai activation
@torch.library.custom_op(
    "rbln_custom_ops::custom_moe_glu_mxfp4",
    mutates_args=(),
)
def custom_moe_glu_mxfp4(
    hidden_states: torch.Tensor,
    gate_proj_blocks: torch.Tensor,
    gate_proj_scales: torch.Tensor,
    gate_proj_bias: torch.Tensor,
    up_proj_blocks: torch.Tensor,
    up_proj_scales: torch.Tensor,
    up_proj_bias: torch.Tensor,
    down_proj_blocks: torch.Tensor,
    down_proj_scales: torch.Tensor,
    down_proj_bias: torch.Tensor,
    router_logits: torch.Tensor,
    alpha: torch.Tensor,
    limit: torch.Tensor,
    k: int,
    post_norm: bool = True,
    expert_map: torch.Tensor | None = None,
    dp_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    MoE GLU operation for GPT-OSS with mxfp4 quantization and swigluoai activation.

    Expected tensor shapes:
    - hidden_states: [num_tokens, hidden_size]
    - gate_proj_blocks: uint8 [num_experts, intermediate_size, hidden_size // 2]
    - gate_proj_scales: [num_experts, intermediate_size, hidden_size // 32]
    - gate_proj_bias: [num_experts, intermediate_size]
    - up_proj_blocks: uint8 [num_experts, intermediate_size, hidden_size // 2]
    - up_proj_scales: [num_experts, intermediate_size, hidden_size // 32]
    - up_proj_bias: [num_experts, intermediate_size]
    - down_proj_blocks: uint8 [num_experts, hidden_size, intermediate_size // 2]
    - down_proj_scales: [num_experts, hidden_size, intermediate_size // 32]
    - down_proj_bias: [num_experts, hidden_size]
    - router_logits: [num_tokens, num_experts]
    - alpha: [], constant
    - limit: [], constant
    - expert_map: [num_experts],
      Mapping from global expert index to local expert index (in num_experts).
      Contains -1 for experts not assigned to the current rank.

    Returns:
        torch.Tensor: [num_tokens, hidden_size]
    """

    if envs.VLLM_RBLN_COMPILE_MODEL:
        return torch.empty_like(hidden_states)

    # Reference torch native implementation

    num_tokens, hidden_size = hidden_states.shape
    num_local_experts = gate_proj_blocks.shape[0]
    # num_global_experts = router_logits.shape[1]
    dtype = hidden_states.dtype

    alpha_val = alpha.item()
    limit_val = limit.item()

    # Compute top-k routing
    # router_logits: [num_tokens, num_global_experts]
    top_k_values, top_k_indices = torch.topk(router_logits, k, dim=-1)

    # Apply softmax to get routing weights (only over selected experts)
    if post_norm:
        routing_weights = torch.softmax(top_k_values, dim=-1)
    else:
        # Pre-norm: softmax over all experts, then select top-k
        all_weights = torch.softmax(router_logits, dim=-1)
        routing_weights = torch.gather(all_weights, dim=-1, index=top_k_indices)

    # Initialize output
    output = torch.zeros(num_tokens, hidden_size, dtype=dtype)

    # Dequantize all expert weights once
    # gate_proj: [num_local_experts, intermediate_size, hidden_size]
    gate_proj_weights = _dequantize_mxfp4(
        gate_proj_blocks, gate_proj_scales, dtype=dtype
    )
    up_proj_weights = _dequantize_mxfp4(up_proj_blocks, up_proj_scales, dtype=dtype)
    down_proj_weights = _dequantize_mxfp4(
        down_proj_blocks, down_proj_scales, dtype=dtype
    )

    # Process each local expert
    for local_expert_idx in range(num_local_experts):
        # Determine which global expert this local expert corresponds to
        if expert_map is not None:
            # Find global expert index that maps to this local expert
            global_expert_idx = (expert_map == local_expert_idx).nonzero(as_tuple=True)[
                0
            ]
            if len(global_expert_idx) == 0:
                continue
            global_expert_idx = global_expert_idx[0].item()
        else:
            global_expert_idx = local_expert_idx

        # Find tokens routed to this expert
        # top_k_indices: [num_tokens, k]
        expert_mask = top_k_indices == global_expert_idx  # [num_tokens, k]
        token_indices, k_indices = expert_mask.nonzero(as_tuple=True)

        if len(token_indices) == 0:
            continue

        # Get routing weights for these tokens
        weights = routing_weights[token_indices, k_indices]  # [num_selected_tokens]

        # Get hidden states for selected tokens
        selected_hidden = hidden_states[token_indices]  # [num_selected, hidden_size]

        # Get expert weights
        gate_w = gate_proj_weights[local_expert_idx]
        gate_b = gate_proj_bias[local_expert_idx]
        up_w = up_proj_weights[local_expert_idx]
        up_b = up_proj_bias[local_expert_idx]
        down_w = down_proj_weights[local_expert_idx]
        down_b = down_proj_bias[local_expert_idx]

        # Forward pass through expert MLP
        gate = selected_hidden @ gate_w.T + gate_b
        up = selected_hidden @ up_w.T + up_b
        activated = _swigluoai(gate, up, alpha_val, limit_val)
        expert_out = activated @ down_w.T + down_b  # [num_selected, hidden_size]

        # Apply routing weights and accumulate
        weighted_out = expert_out * weights.unsqueeze(-1)
        output.index_add_(0, token_indices, weighted_out.to(dtype))

    return output


@custom_moe_glu_mxfp4.register_fake
def custom_moe_glu_mxfp4_fake(
    hidden_states: torch.Tensor,
    gate_proj_blocks: torch.Tensor,
    gate_proj_scales: torch.Tensor,
    gate_proj_bias: torch.Tensor,
    up_proj_blocks: torch.Tensor,
    up_proj_scales: torch.Tensor,
    up_proj_bias: torch.Tensor,
    down_proj_blocks: torch.Tensor,
    down_proj_scales: torch.Tensor,
    down_proj_bias: torch.Tensor,
    router_logits: torch.Tensor,
    alpha: torch.Tensor,
    limit: torch.Tensor,
    k: int,
    post_norm: bool = True,
    expert_map: torch.Tensor | None = None,
    dp_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)
