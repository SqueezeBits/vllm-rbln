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

# NOTE(RBLN): This patch originated from https://github.com/RBLN-SW/vllm-rbln/pull/145
# and its active v0.12 forward variant was updated in
# https://github.com/RBLN-SW/vllm-rbln/commit/fd0f28fd60042a93ef4deff0a5f99cc28ffb0643.

from collections.abc import Iterable

import torch
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from vllm.model_executor.models.utils import is_pp_missing_parameter

logger = init_logger(__name__)


def patched_qwen2_moe_sparse_moe_block_forward(
    self: Qwen2MoeSparseMoeBlock,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # NOTE(RBLN): fused-MoE keeps the original token layout; do not flatten here.
    final_hidden_states = self.experts(
        hidden_states=hidden_states,
        router=lambda x: self.gate(x)[0],
    )
    if self.shared_expert is not None:
        final_hidden_states = final_hidden_states[0] + final_hidden_states[1]
    if self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states
        )
    return final_hidden_states


# NOTE(RBLN): This load-weights patch originated from
# https://github.com/RBLN-SW/vllm-rbln/commit/d6c5ec8960a6108e94698b71191e12e887c09184
# and was later expanded in
# https://github.com/RBLN-SW/vllm-rbln/pull/81,
# https://github.com/RBLN-SW/vllm-rbln/pull/435, and
# https://github.com/RBLN-SW/vllm-rbln/pull/511.


def patched_qwen2_moe_load_weights(
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
            # Skip loading extra bias for GPTQ models.
            if (
                name.endswith(".bias") or name.endswith("_bias")
            ) and name not in params_dict:
                continue
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        logger.warning_once(
                            "Found kv_scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv_scale is not loaded.",  #  noqa: E501
                            name,
                            remapped_kv_scale_name,
                        )
                        continue
                    else:
                        name = remapped_kv_scale_name
                # GGUF: make sure that shared_expert_gate is a 2D tensor.
                if "mlp.shared_expert_gate" in name and len(loaded_weight.shape) == 1:
                    loaded_weight = loaded_weight[None, :]
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params
