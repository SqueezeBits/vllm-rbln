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

# NOTE(RBLN): This load-weights patch originated from
# https://github.com/RBLN-SW/vllm-rbln/commit/d6c5ec8960a6108e94698b71191e12e887c09184
# and was later expanded in
# https://github.com/RBLN-SW/vllm-rbln/pull/81,
# https://github.com/RBLN-SW/vllm-rbln/pull/435, and
# https://github.com/RBLN-SW/vllm-rbln/pull/511.

from collections.abc import Iterable

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import is_pp_missing_parameter


def patched_llama4_load_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    fused_experts_params = False
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.num_experts,
    )
    expert_params_mapping_fused = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_up_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="gate_up_proj",
        num_experts=1,
    )
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
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

        if "experts.gate_up_proj" in name or "experts.down_proj" in name:
            fused_experts_params = True
            expert_params_mapping = expert_params_mapping_fused
        if self.quant_config is not None and (
            scale_name := self.quant_config.get_cache_scale(name)
        ):
            # Loading kv cache quantization scales
            param = params_dict[scale_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            loaded_weight = (
                loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
            )
            weight_loader(param, loaded_weight)
            loaded_params.add(scale_name)
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name or "experts" in name:
                continue
            name = name.replace(weight_name, param_name)
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(name)
            break
        else:
            moe_loaded = self.load_moe_expert_weights(
                name,
                loaded_weight,
                params_dict,
                loaded_params,
                expert_params_mapping,
                fused=fused_experts_params,
            )

            if not moe_loaded:
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
    return loaded_params
