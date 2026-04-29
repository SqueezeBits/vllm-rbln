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

# NOTE(RBLN): This patch originated from the MoE adaptation series including
# https://github.com/RBLN-SW/vllm-rbln/commit/27f4337,
# https://github.com/RBLN-SW/vllm-rbln/commit/7fac24d,
# https://github.com/RBLN-SW/vllm-rbln/commit/1b57867,
# https://github.com/RBLN-SW/vllm-rbln/commit/6278c7c,
# https://github.com/RBLN-SW/vllm-rbln/commit/a99eff3, and
# https://github.com/RBLN-SW/vllm-rbln/commit/fc530d1.

from collections.abc import Callable

import torch
import torch.nn.functional as F
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.layers.fused_moe.utils import (
    get_masked_routing_weights,
    get_tokens_mask,
)

logger = init_logger(__name__)

_UPSTREAM_FUSED_MOE_INIT = FusedMoE.__init__

if envs.VLLM_RBLN_MOE_USE_OPT_KERNEL:

    @torch.library.custom_op(
        "rbln_custom_ops::custom_moe_glu",
        mutates_args=(),
    )
    def custom_moe_glu(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        topk: int,
        post_norm: bool,
        expert_map: torch.Tensor | None = None,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
        dp_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        for i in range(expert_cnt):
            gate = F.linear(hidden_states, gate_proj_weight[i])
            up = F.linear(hidden_states, up_proj_weight[i])
            mul = F.silu(gate) * up
            down = F.linear(mul, down_proj_weight[i])
            out += down * masked_routing_weight[:, i : i + 1]
        return out

    @custom_moe_glu.register_fake
    def custom_moe_glu_fake(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        topk: int,
        post_norm: bool,
        expert_map: torch.Tensor | None = None,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
        dp_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)

else:

    @torch.library.custom_op(
        "rbln_custom_ops::custom_moe_glu",
        mutates_args=(),
    )
    def custom_moe_glu(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        expert_select_count: torch.Tensor,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
        dp_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = torch.zeros_like(hidden_states)
        expert_cnt = gate_proj_weight.shape[0]
        for i in range(expert_cnt):
            gate = F.linear(hidden_states, gate_proj_weight[i])
            up = F.linear(hidden_states, up_proj_weight[i])
            mul = F.silu(gate) * up
            down = F.linear(mul, down_proj_weight[i])
            out += down * masked_routing_weight[:, i : i + 1]
        return out

    @custom_moe_glu.register_fake
    def custom_moe_glu_fake(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        masked_routing_weight: torch.Tensor,
        expert_select_count: torch.Tensor,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,
        dp_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)


def patched_fused_moe_init(self: FusedMoE, *args, **kwargs) -> None:
    _UPSTREAM_FUSED_MOE_INIT(self, *args, **kwargs)
    self.expert_map_const = (
        self.expert_map.tolist() if self.expert_map is not None else None
    )


def _unquantized_fused_moe_method_rbln(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    del self
    w1 = layer.w13_weight
    w2 = layer.w2_weight

    orig_shape = x.shape
    hidden_size = x.shape[-1]
    num_tokens = x.shape[:-1].numel()
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    dtype = x.dtype
    top_k = layer.top_k

    hidden_states = x
    gating_output = router_logits
    topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    topk_weights = topk_weights.to(torch.float)
    topk_weights, selected_experts = topk_weights.topk(top_k, dim=-1)
    if layer.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if layer.expert_map is not None:
        selected_experts = layer.expert_map[selected_experts]

    final_hidden_states = None

    hidden_states = hidden_states.reshape(1, num_tokens, -1)
    expert_idx_array = torch.arange(0, num_experts).reshape(num_experts, 1, 1, 1)
    selected_experts_array = selected_experts.reshape(-1, 1, num_tokens, top_k)
    expert_mask_array = selected_experts_array == expert_idx_array
    topk_weights_array = topk_weights.reshape(-1, 1, num_tokens, top_k)
    expert_weights_array = (topk_weights_array * expert_mask_array).sum(
        dim=-1, keepdim=True
    )
    temp_expert_weights = expert_weights_array[0]
    hidden_states = hidden_states + temp_expert_weights - temp_expert_weights
    hidden_states = hidden_states.to(dtype)
    expert_weights_array = expert_weights_array.broadcast_to(
        (num_experts, 1, num_tokens, hidden_size)
    ).to(dtype)

    for expert_idx in range(num_experts):
        expert_w1 = w1[expert_idx]
        expert_w2 = w2[expert_idx]
        expert_weights = expert_weights_array[expert_idx]
        x = F.linear(hidden_states, expert_w1)
        gate = F.silu(x[..., :intermediate_size])
        x = x[..., intermediate_size:] * gate
        x = F.linear(x, expert_w2)

        current_hidden_states = x * expert_weights
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states = final_hidden_states + current_hidden_states

    assert final_hidden_states is not None
    return final_hidden_states.reshape(orig_shape)


def _unquantized_fused_moe_method_custom(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    del self
    orig_shape = x.shape
    num_tokens = orig_shape[:-1].numel()
    intermediate_size = layer.w2_weight.shape[-1]

    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
    down_proj_weight = layer.w2_weight

    hidden_states = x.reshape(num_tokens, -1)
    router_logits = router_logits.reshape(num_tokens, -1)

    masked_routing_weights, expert_select_count = get_masked_routing_weights(
        router_logits, layer.top_k, layer.renormalize, layer.expert_map
    )

    tokens_mask = None
    if envs.VLLM_RBLN_USE_MOE_TOKENS_MASK:
        tokens_mask = get_tokens_mask(num_tokens)

    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        masked_routing_weights,
        expert_select_count,
        None,
        None,
        None,
        tokens_mask,
    )
    return final_hidden_states.reshape(orig_shape)


def _unquantized_fused_optimize_moe_method_custom(
    self: UnquantizedFusedMoEMethod,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    del self
    orig_shape = x.shape
    num_tokens = orig_shape[:-1].numel()
    intermediate_size = layer.w2_weight.shape[-1]

    gate_proj_weight = layer.w13_weight[:, :intermediate_size, :]
    up_proj_weight = layer.w13_weight[:, intermediate_size:, :]
    down_proj_weight = layer.w2_weight

    hidden_states = x.reshape(num_tokens, -1)
    router_logits = router_logits.reshape(num_tokens, -1)

    expert_map_const = None
    if layer.expert_map is not None:
        assert getattr(layer, "expert_map_const", None) is not None
        expert_map_const = torch.tensor(layer.expert_map_const, dtype=torch.int32)

    tokens_mask = None
    if envs.VLLM_RBLN_USE_MOE_TOKENS_MASK:
        tokens_mask = get_tokens_mask(num_tokens)

    final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        router_logits,
        layer.top_k,
        layer.renormalize,
        expert_map_const,
        None,
        None,
        None,
        tokens_mask,
    )
    return final_hidden_states.reshape(orig_shape)


def _select_unquantized_apply() -> Callable[
    [UnquantizedFusedMoEMethod, FusedMoE, torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    if envs.VLLM_RBLN_MOE_USE_OPT_KERNEL:
        logger.info("[RBLN] fused moe, RBLN optimize moe custom kernel")
        return _unquantized_fused_optimize_moe_method_custom
    if envs.VLLM_RBLN_MOE_CUSTOM_KERNEL:
        logger.info("[RBLN] fused moe, RBLN moe custom kernel")
        return _unquantized_fused_moe_method_custom

    logger.info("[RBLN] fused moe, pytorch native kernel")
    return _unquantized_fused_moe_method_rbln


def patched_fused_moe_forward_oot(
    self: FusedMoE,
    hidden_states: torch.Tensor,
    router: torch.nn.Module,
) -> torch.Tensor:
    assert self.quant_method is not None

    if self.moe_parallel_config.dp_size > 1:
        org_hidden_shape = hidden_states.shape
        hidden_states = self.naive_multicast(hidden_states)
    router_logits = router(hidden_states)

    final_hidden_states = self.quant_method.apply(
        layer=self,
        x=hidden_states,
        router_logits=router_logits,
    )

    if self.moe_parallel_config.dp_size > 1:
        if envs.VLLM_RBLN_MOE_REDUCE_SCATTER:
            hidden_shape_dp = (-1, 1, org_hidden_shape[-1])
            all_hidden_states = final_hidden_states.reshape(hidden_shape_dp)
            assert all_hidden_states.shape[0] % self.moe_parallel_config.dp_size == 0

            hidden_states = get_dp_group().reduce_scatter(all_hidden_states, dim=0)
            max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
            assert hidden_states.shape[0] == max_pad

            num_tokens = org_hidden_shape[:-1].numel()
            final_hidden_states = hidden_states[:num_tokens]
        else:
            all_hidden_states = get_dp_group().all_reduce(final_hidden_states)
            hidden_shape_dp = (-1, 1, org_hidden_shape[-1])
            final_hidden_states = all_hidden_states.reshape(hidden_shape_dp)

            max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
            num_tokens = org_hidden_shape[:-1].numel()
            start = self.moe_parallel_config.dp_rank * max_pad
            end = start + num_tokens
            final_hidden_states = final_hidden_states[start:end]

        final_hidden_states = final_hidden_states.reshape(org_hidden_shape)

    return final_hidden_states


def patched_fused_moe_naive_multicast(self: FusedMoE, x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(1, -1, x.size(-1))
    max_pad = get_forward_context().dp_metadata.max_pads_across_dp.shape[0]
    num_tokens = x.size(1)
    num_repeat = max_pad // num_tokens
    x = x.repeat(num_repeat, 1, 1)
    x = x.reshape(1, max_pad, -1)

    if not envs.VLLM_RBLN_DP_INPUT_ALL_GATHER:
        all_buffer = None
        zeros = x - x
        for rank in range(get_dp_group().world_size):
            rank_tensor = x if rank == self.moe_parallel_config.dp_rank else zeros
            all_buffer = (
                torch.cat((all_buffer, rank_tensor), dim=0)
                if all_buffer is not None
                else rank_tensor
            )
        return get_dp_group().all_reduce(all_buffer)

    return get_dp_group().all_gather(x, dim=0)


patched_unquantized_fused_moe_method_apply = _select_unquantized_apply()
