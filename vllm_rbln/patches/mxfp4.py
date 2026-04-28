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

# NOTE(RBLN): This patch originated from
# https://github.com/RBLN-SW/vllm-rbln/commit/9db13b2 while adding RBLN
# Mxfp4MoEMethod support, and was later updated through
# https://github.com/RBLN-SW/vllm-rbln/pull/293,
# https://github.com/RBLN-SW/vllm-rbln/pull/330,
# https://github.com/RBLN-SW/vllm-rbln/pull/369,
# https://github.com/RBLN-SW/vllm-rbln/pull/435,
# https://github.com/RBLN-SW/vllm-rbln/pull/438,
# https://github.com/RBLN-SW/vllm-rbln/pull/511, and
# https://github.com/RBLN-SW/vllm-rbln/pull/531.

import torch
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.utils import set_weight_attrs

import vllm_rbln.rbln_envs as envs
from vllm_rbln.patches.patch_registry import register_patch


@register_patch(
    target="vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod",
    reason=(
        "RBLN needs a custom MXFP4 MoE method because upstream uses "
        "Mxfp4MoEMethod for non-XPU platforms but does not expose an "
        "out-of-tree kernel interface."
    ),
)
class Mxfp4MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.moe = moe

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        # swigluoai constant value
        # gemm1_alpha = 1.702, gemm1_beta = 1.0, gemm1_clamp_limit = 7.0
        # gemm1_alpha = 1.702
        self.swiglu_alpha = torch.tensor(1.702, dtype=torch.float32)
        # gemm1_clamp_limit = 7.0
        self.swiglu_limit = torch.tensor(7.0, dtype=torch.float32)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert isinstance(layer, FusedMoE)

        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        mxfp4_block = 32

        intermediate_size_per_partition_after_pad = intermediate_size_per_partition

        # NOTE: upstream rounds up intermediate_size_per_partition/hidden_size
        assert intermediate_size_per_partition % 64 == 0

        self.intermediate_size = intermediate_size_per_partition_after_pad
        self.hidden_size = hidden_size
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        assert isinstance(layer, FusedMoE)

        # w1
        layer.register_buffer("gate_proj_blocks", layer.w13_weight.data[:, ::2])
        layer.register_buffer("gate_proj_scales", layer.w13_weight_scale.data[:, ::2])
        layer.register_buffer("gate_proj_bias", layer.w13_bias.data[:, ::2])

        # w3
        layer.register_buffer("up_proj_blocks", layer.w13_weight.data[:, 1::2])
        layer.register_buffer("up_proj_scales", layer.w13_weight_scale.data[:, 1::2])
        layer.register_buffer("up_proj_bias", layer.w13_bias.data[:, 1::2])

        # w2
        layer.register_buffer("down_proj_blocks", layer.w2_weight.data)
        layer.register_buffer("down_proj_scales", layer.w2_weight_scale.data)
        layer.register_buffer("down_proj_bias", layer.w2_bias.data)

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular:
        # NOTE(RBLN): this is used only for "modular kernel"
        raise NotImplementedError()

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # NOTE(RBLN): this is used only for "modular kernel"
        raise NotImplementedError

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # refer to custom_moe_glu
        orig_shape = x.shape  # noqa: F841
        num_tokens = orig_shape[:-1].numel()  # noqa: F841
        hidden_states = x.reshape(num_tokens, -1)
        router_logits = router_logits.reshape(num_tokens, -1)
        # x = x.view(-1, self.hidden_size)
        # router_logits = router_logits.view(-1, self.num_experts)
        # router_logits = router_logits.view(-1, self.moe.num_experts)

        if layer.activation == MoEActivation.SWIGLUOAI:
            expert_map_const = None
            if layer.expert_map is not None:
                assert getattr(layer, "expert_map_const", None) is not None
                expert_map_const = torch.tensor(
                    layer.expert_map_const, dtype=torch.int32
                )

            tokens_mask = None
            use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
            if use_moe_tokens_mask:
                from vllm_rbln.model_executor.layers.fused_moe.layer import (
                    get_tokens_mask,
                )

                tokens_mask = get_tokens_mask(num_tokens)

            final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu_mxfp4(
                hidden_states,
                layer.gate_proj_blocks,
                layer.gate_proj_scales,
                layer.gate_proj_bias,
                layer.up_proj_blocks,
                layer.up_proj_scales,
                layer.up_proj_bias,
                layer.down_proj_blocks,
                layer.down_proj_scales,
                layer.down_proj_bias,
                router_logits,
                self.swiglu_alpha,
                self.swiglu_limit,
                layer.top_k,
                layer.renormalize,
                expert_map_const,
                tokens_mask,
            )
        else:
            raise NotImplementedError(layer.activation)

        return final_hidden_states.reshape(orig_shape)
