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
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2MoE,
)
from vllm.model_executor.models.utils import is_pp_missing_parameter


def patched_deepseek_v2_load_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.n_routed_experts,
    )

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        """
        [RBLN] Skips loading of layers greater than `num_hidden_layers`.
        This must be modified to more graceful code in the future.
        """
        if name.startswith("model.layers"):
            layer_idx = int(name.split(".")[2])
            if layer_idx >= self.config.num_hidden_layers:
                continue
        #######
        if "rotary_emb.inv_freq" in name:
            continue

        spec_layer = deepseek_v2.get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is not None:
            continue  # skip spec decode layers for main model

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
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, self):
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

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id
                )
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


# NOTE(RBLN): This DeepSeekV2 forward patch originated from
# https://github.com/RBLN-SW/vllm-rbln/commit/d6c5ec8960a6108e94698b71191e12e887c09184
# and was later adjusted in
# https://github.com/RBLN-SW/vllm-rbln/pull/81 and
# https://github.com/RBLN-SW/vllm-rbln/pull/367.
def patched_deepseek_v2_moe_forward(
    self: DeepseekV2MoE,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    shared_output = None
    if self.n_shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)
    if hidden_states.dtype != torch.float16:
        final_hidden_states = (
            self.experts(
                hidden_states=hidden_states,
                router=lambda x: self.gate(x)[0],
            )
            * self.routed_scaling_factor
        )
    else:
        # Fix FP16 overflow. See DeepseekV2DecoderLayer for more details.
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router=lambda x: self.gate(x)[0],
        )
    if shared_output is not None:
        if hidden_states.dtype != torch.float16:
            final_hidden_states = final_hidden_states + shared_output
        else:
            final_hidden_states = final_hidden_states + shared_output * (
                1.0 / self.routed_scaling_factor
            )
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states


def patched_deepseek_v2_attention_forward(
    self: DeepseekV2Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    batch, _, _ = hidden_states.shape
    if self.q_lora_rank is not None:
        q = self.q_a_proj(hidden_states)[0]
        q = self.q_a_layernorm(q)
        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
    else:
        q = self.q_proj(hidden_states)[0].view(
            -1, self.num_local_heads, self.qk_head_dim
        )
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
    kv_a, k_pe = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_a = self.kv_a_layernorm(kv_a.contiguous())
    kv = self.kv_b_proj(kv_a)[0]
    kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)

    q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
    if q_nope.dim() != q_pe.dim():
        q_pe = q_pe.squeeze(0)
    if k_nope.dim() != k_pe.dim():
        k_pe = k_pe.squeeze(0)

    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.repeat(1, self.num_local_heads, 1)], dim=-1)
    if self.qk_head_dim != self.v_head_dim:
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim], value=0
        ).view(-1, self.num_local_heads * self.qk_head_dim)
    q = q.reshape(batch, -1, self.num_local_heads * self.qk_head_dim)
    k = k.reshape(batch, -1, self.num_local_heads * self.qk_head_dim)
    v = v.reshape(batch, -1, self.num_local_heads * self.qk_head_dim)
    attn_output = self.attn(q, k, v)
    if self.qk_head_dim != self.v_head_dim:
        attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)[
            ..., : self.v_head_dim
        ].reshape(batch, -1, self.num_local_heads * self.v_head_dim)

    output, _ = self.o_proj(attn_output)
    return output
