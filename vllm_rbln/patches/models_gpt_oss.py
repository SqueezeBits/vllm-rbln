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

from collections.abc import Iterable

import torch
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gpt_oss import GptOssModel, MLPBlock
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.utils.math_utils import cdiv

from vllm_rbln.patches.patch_registry import register_patch

# NOTE(RBLN): This patch originated from
# https://github.com/RBLN-SW/vllm-rbln/commit/a036d1241baa79a4589b7e6ad1625f3458e09210
# and was later updated in
# https://github.com/RBLN-SW/vllm-rbln/pull/293 and
# https://github.com/RBLN-SW/vllm-rbln/pull/367.


@register_patch(
    target="vllm.model_executor.models.gpt_oss.GptOssModel._load_weights_mxfp4",
    reason=(
        "The RBLN path needs GPT-OSS MXFP4 weight loading to honor "
        "num_hidden_layers overrides while preserving the expert-weight "
        "slicing rules used by the RBLN MoE path, because the upstream "
        "loader assumes the full upstream layer set during weight traversal."
    ),
)
def rbln_gpt_oss_load_weights_mxfp4(
    self: GptOssModel,
    ep_rank_end: int,
    ep_rank_start: int,
    heads_per_rank: int,
    head_start: int,
    weights: Iterable[tuple[str, torch.Tensor]],
    stacked_params_mapping: list[tuple[str, ...]],
) -> set[str]:
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    mxfp4_block = 32
    use_ep = self.parallel_config.enable_expert_parallel
    num_experts = self.config.num_local_experts

    tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
        tp_size=get_tensor_model_parallel_world_size(),
        dp_size=get_dp_group().world_size,
        dp_rank=get_dp_group().rank_in_group,
        pcp_size=get_pcp_group().world_size,
        pcp_rank=get_pcp_group().rank_in_group,
    )

    intermediate_size = self.config.intermediate_size
    intermediate_size_block = intermediate_size // mxfp4_block
    per_rank_intermediate_size_block = cdiv(intermediate_size_block, tp_size)
    per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block

    tp_rank_start = tp_rank * per_rank_intermediate_size
    tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

    for name, weight in weights:
        if name.startswith("layers"):
            layer_idx = int(name.split(".")[1])
            if layer_idx >= self.config.num_hidden_layers:
                continue

        if is_pp_missing_parameter(name, self):
            continue

        if ".w13_weight_scale" in name:
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                narrow_weight,
                weight_name=name,
                shard_id=None,
                expert_id=None,
            )
            loaded_params.add(name)
            continue
        elif ".w2_weight_scale" in name:
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[
                    ..., tp_rank_start // mxfp4_block : tp_rank_end // mxfp4_block
                ]

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                narrow_weight,
                weight_name=name,
                shard_id=None,
                expert_id=None,
            )
            loaded_params.add(name)
            continue
        elif ".w13_weight" in name:
            weight = weight.view(num_experts, 2 * intermediate_size, -1).contiguous()

            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                narrow_weight,
                weight_name=name,
                shard_id=None,
                expert_id=None,
            )
            loaded_params.add(name)
            continue
        elif ".w2_weight" in name:
            weight = weight.view(num_experts, -1, intermediate_size // 2).contiguous()
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[..., tp_rank_start // 2 : tp_rank_end // 2]

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                narrow_weight,
                weight_name=name,
                shard_id=None,
                expert_id=None,
            )
            loaded_params.add(name)
            continue
        elif ".w13_bias" in name:
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                narrow_weight,
                weight_name=name,
                shard_id=None,
                expert_id=None,
            )
            loaded_params.add(name)
            continue
        elif ".w2_bias" in name:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if use_ep:
                weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                if tp_rank != 0:
                    weight.zero_()
            weight_loader(
                param, weight, weight_name=name, shard_id=None, expert_id=None
            )
            loaded_params.add(name)
            continue
        elif "sinks" in name:
            param = params_dict[name]
            narrow_weight = weight.narrow(0, head_start, heads_per_rank)
            param.data.copy_(narrow_weight)
            loaded_params.add(name)
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if weight_loader == default_weight_loader:
                weight_loader(param, weight)
            else:
                weight_loader(param, weight, shard_id)
            break
        else:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
        loaded_params.add(name)
    return loaded_params


@register_patch(
    target="vllm.model_executor.models.gpt_oss.MLPBlock.forward",
    reason=(
        "The RBLN path needs GPT-OSS MoE blocks to route experts through the "
        "patched router callback and apply an explicit tensor-parallel "
        "all-reduce, because the upstream forward path uses "
        "sequence-parallel chunking and router_logits precomputation that do "
        "not match the RBLN fused-MoE execution contract."
    ),
)
def rbln_gpt_oss_mlp_block_forward(
    self: MLPBlock,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    final_hidden_states = self.experts(
        hidden_states=hidden_states,
        router=self.router,
    )
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states
