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

# NOTE(RBLN): This patch originated from
# https://github.com/RBLN-SW/vllm-rbln/commit/d6c5ec8960a6108e94698b71191e12e887c09184
# and its active router/shared-output behavior was updated in
# https://github.com/RBLN-SW/vllm-rbln/pull/367 and
# https://github.com/RBLN-SW/vllm-rbln/pull/511.

import typing
from collections.abc import Callable, Iterable

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from vllm.model_executor.models.utils import is_pp_missing_parameter


def patched_qwen3_moe_sparse_moe_block_forward(
    self: Qwen3MoeSparseMoeBlock,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # RBLN fused-MoE keeps the original token layout and expects router
    # computation to happen inside the patched expert path.
    shared_out, fused_out = self.experts(
        hidden_states=hidden_states,
        router=lambda x: self.gate(x)[0],
    )
    final_hidden_states = (
        shared_out + fused_out if shared_out is not None else fused_out
    )
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states


# NOTE(RBLN): This load-weights patch originated from
# https://github.com/RBLN-SW/vllm-rbln/commit/d6c5ec8960a6108e94698b71191e12e887c09184
# and was later expanded in
# https://github.com/RBLN-SW/vllm-rbln/pull/81,
# https://github.com/RBLN-SW/vllm-rbln/pull/435, and
# https://github.com/RBLN-SW/vllm-rbln/pull/511.


def patched_qwen3_moe_load_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # Skip loading extra parameters for GPTQ/modelopt models.
    ignore_suffixes = (
        ".bias",
        "_bias",
        ".weight_scale",
        "_weight_scale",
        ".input_scale",
        "_input_scale",
    )

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    expert_params_mapping = self.get_expert_mapping()
    for name, loaded_weight in weights:
        """
        [RBLN] Skips loading of layers greater than `num_hidden_layers`.
        This must be modified to more graceful code in the future.
        """
        if name.startswith("layers"):
            layer_idx = int(name.split(".")[1])
            if layer_idx >= self.config.num_hidden_layers:
                continue
        #######

        if self.quant_config is not None and (
            scale_name := self.quant_config.get_cache_scale(name)
        ):
            # Loading kv cache quantization scales
            param = params_dict[scale_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            assert loaded_weight.numel() == 1, (
                f"KV scale numel {loaded_weight.numel()} != 1"
            )
            loaded_weight = loaded_weight.squeeze()
            weight_loader(param, loaded_weight)
            loaded_params.add(scale_name)
            continue
        if "scale" in name or "zero_point" in name:
            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if "mlp.experts" in name:
                continue
            name = name.replace(weight_name, param_name)

            # Skip loading extra parameters for GPTQ/modelopt models.
            if name.endswith(ignore_suffixes) and name not in params_dict:
                continue

            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue
            if name.endswith("scale"):
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if weight_loader == default_weight_loader:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)
            break
        else:
            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue

                # Anyway, this is an expert weight and should not be
                # attempted to load as other weights later
                is_expert_weight = True

                # Do not modify `name` since the loop may continue here
                # Instead, create a new variable
                name_mapped = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name_mapped, self):
                    continue

                # Skip loading extra parameters for GPTQ/modelopt models.
                if (
                    name_mapped.endswith(ignore_suffixes)
                    and name_mapped not in params_dict
                ):
                    continue

                param = params_dict[name_mapped]
                # We should ask the weight loader to return success or not
                # here since otherwise we may skip experts with other
                # available replicas.
                weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    name = name_mapped
                    break
            else:
                if is_expert_weight:
                    # We've checked that this is an expert weight
                    # However it's not mapped locally to this rank
                    # So we simply skip it
                    continue

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params
