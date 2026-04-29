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
from vllm.forward_context import get_forward_context

import vllm_rbln.rbln_envs as envs


def get_tokens_mask(
    num_tokens: int,
    left: float = 1.0,
    right: float = 0.0,
) -> torch.Tensor:
    num_tokens_across_dp = get_forward_context().dp_metadata.num_tokens_across_dp_cpu
    num_tokens_across_dp = num_tokens_across_dp.unsqueeze(1)
    if num_tokens_across_dp.size(0) == 1:
        max_pad = num_tokens
    else:
        max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
    pos = torch.arange(max_pad, dtype=torch.int32).unsqueeze(0)
    tokens_mask = torch.where(pos < num_tokens_across_dp, left, right)
    return tokens_mask.reshape(-1, 1)


def get_masked_routing_weights(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    expert_map: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if renormalize:
        router_logits = router_logits.to(torch.float)
        selected_weights, selected_experts = torch.topk(router_logits, k=top_k, dim=-1)
        selected_weights = torch.nn.functional.softmax(selected_weights, dim=1)
    else:
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1)
        routing_weights = routing_weights.to(torch.float)
        selected_weights, selected_experts = torch.topk(
            routing_weights, k=top_k, dim=-1
        )

    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if use_moe_tokens_mask:
        tokens_mask = get_tokens_mask(router_logits.shape[0], 1.0, 0.0)
        selected_weights = selected_weights * tokens_mask

    n_expert = router_logits.shape[1]
    if expert_map is not None:
        expert_map_within_bounds = torch.where(
            expert_map < 0, n_expert - 1, expert_map
        ).to(torch.int64)
        selected_experts = expert_map_within_bounds[selected_experts]

    masked_routing_weights = torch.zeros_like(router_logits, dtype=torch.float32)
    masked_routing_weights.scatter_(1, selected_experts, selected_weights)

    zeros = torch.zeros(n_expert, dtype=torch.int32)

    if use_moe_tokens_mask:
        ones = torch.ones_like(selected_experts, dtype=torch.int32)
        tokens_mask = tokens_mask.to(torch.int32)
        ones = ones * tokens_mask
        ones = ones.view(-1)
    else:
        ones = torch.ones_like(selected_experts.view(-1), dtype=torch.int32)

    expert_select_count = torch.scatter_add(
        zeros, dim=0, index=selected_experts.view(-1), src=ones
    )

    return masked_routing_weights, expert_select_count
