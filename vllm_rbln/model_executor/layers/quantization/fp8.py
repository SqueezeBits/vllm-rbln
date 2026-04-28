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


@torch.library.custom_op(
    "rbln_custom_ops::custom_moe_swiglu_group_dequantize",
    mutates_args=(),
)
def custom_moe_swiglu_group_dequantize(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    gate_proj_scale: torch.Tensor,
    up_proj_weight: torch.Tensor,
    up_proj_scale: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_scale: torch.Tensor,
    router_logits: torch.Tensor,
    group_size: torch.Tensor,
    topk: int,
    e_score_correction_bias: torch.Tensor | None = None,
    gate_proj_bias: torch.Tensor | None = None,
    up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    dp_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Customized MoE SwiGLU operation.

    Expected tensor shapes:
    - hidden_states: [batch*seq_len, hidden_size]
    - gate_proj_weight: [num_experts, hidden_size, intermediate_size]
    - gate_proj_scale: [num_experts, intermediate_size, hidden_size // 128]
    - up_proj_weight: [num_experts, hidden_size, intermediate_size]
    - up_proj_scale: [num_experts, intermediate_size, hidden_size // 128]
    - down_proj_weight: [num_experts, intermediate_size, hidden_size]
    - down_proj_scale: [num_experts, hidden_size, intermediate_size // 128]
    - router_logits: [batch*seq_len, num_experts]
    - group_size: group size for weight scale
    - topk: top k experts to select
    - e_score_correction_bias:
    - gate_proj_bias: [num_experts, intermediate_size]
    - up_proj_bias: [num_experts, intermediate_size]
    - down_proj_bias: [num_experts, hidden_size]

    Returns:
        Tensor: [batch * seq_len, hidden_size]
    """

    def _dequantize_blockwise_weight(
        weight: torch.Tensor,
        scale: torch.Tensor,
        in_block_size: int,
        out_block_size: int | None = None,
    ) -> torch.Tensor:
        # `weight` is [num_experts, out_features, in_features].
        # `scale` is [num_experts, out_blocks, in_blocks].
        out_features = weight.shape[1]
        in_features = weight.shape[2]
        out_blocks = scale.shape[1]
        out_block_size = out_block_size or (
            (out_features + out_blocks - 1) // out_blocks
        )

        expanded = scale.repeat_interleave(out_block_size, dim=1).repeat_interleave(
            in_block_size, dim=2
        )
        expanded = expanded[:, :out_features, :in_features]
        return weight.to(hidden_states.dtype) * expanded.to(hidden_states.dtype)

    in_block_size = int(group_size.item())
    gate_out_block = (
        gate_proj_weight.shape[1] + gate_proj_scale.shape[1] - 1
    ) // gate_proj_scale.shape[1]
    down_out_block = (
        down_proj_weight.shape[1] + down_proj_scale.shape[1] - 1
    ) // down_proj_scale.shape[1]

    gate_proj_weight_dq = _dequantize_blockwise_weight(
        gate_proj_weight, gate_proj_scale, in_block_size, gate_out_block
    )
    up_proj_weight_dq = _dequantize_blockwise_weight(
        up_proj_weight, up_proj_scale, in_block_size, gate_out_block
    )
    down_proj_weight_dq = _dequantize_blockwise_weight(
        down_proj_weight, down_proj_scale, in_block_size, down_out_block
    )

    routing_weights = router_logits.float()
    scores_for_choice = routing_weights
    if e_score_correction_bias is not None:
        scores_for_choice = scores_for_choice + e_score_correction_bias

    _, topk_ids = torch.topk(scores_for_choice, topk, dim=-1, sorted=False)
    topk_weights = routing_weights.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
        1e-20
    )
    topk_weights = topk_weights.to(hidden_states.dtype)

    if dp_mask is not None:
        topk_weights = topk_weights * dp_mask.to(topk_weights.dtype)

    num_experts = gate_proj_weight_dq.shape[0]
    if expert_map is not None:
        safe_expert_map = torch.where(expert_map < 0, num_experts - 1, expert_map).to(
            topk_ids.dtype
        )
        topk_ids = safe_expert_map[topk_ids]

    final_hidden_states = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(
        topk_ids, num_classes=num_experts
    ).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())
        topk_slot, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue

        current_state = hidden_states[token_idx]
        gate = torch.nn.functional.linear(
            current_state,
            gate_proj_weight_dq[expert_idx],
            gate_proj_bias[expert_idx] if gate_proj_bias is not None else None,
        )
        up = torch.nn.functional.linear(
            current_state,
            up_proj_weight_dq[expert_idx],
            up_proj_bias[expert_idx] if up_proj_bias is not None else None,
        )
        swiglu = torch.nn.functional.silu(gate) * up
        down = torch.nn.functional.linear(
            swiglu,
            down_proj_weight_dq[expert_idx],
            down_proj_bias[expert_idx] if down_proj_bias is not None else None,
        )
        current_hidden_states = down * topk_weights[token_idx, topk_slot, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states


@custom_moe_swiglu_group_dequantize.register_fake
def custom_moe_swiglu_group_dequantize_fake(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    gate_proj_scale: torch.Tensor,
    up_proj_weight: torch.Tensor,
    up_proj_scale: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_scale: torch.Tensor,
    router_logits: torch.Tensor,
    group_size: torch.Tensor,
    topk: int,
    e_score_correction_bias: torch.Tensor | None = None,
    gate_proj_bias: torch.Tensor | None = None,
    up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    dp_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)
