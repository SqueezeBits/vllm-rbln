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

from vllm_rbln.model_executor.layers.fused_moe.layer import (
    patched_fused_moe_forward_oot,
    patched_fused_moe_init,
    patched_fused_moe_naive_multicast,
    patched_unquantized_fused_moe_method_apply,
)
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.layers.fused_moe.layer.FusedMoE.__init__",
    reason=(
        "RBLN stores expert_map as a Python constant for custom MoE kernels "
        "while preserving upstream FusedMoE initialization."
    ),
    priority=49,
    owner_module=__name__,
)(patched_fused_moe_init)


register_patch(
    target="vllm.model_executor.layers.fused_moe.layer.FusedMoE.forward_oot",
    reason=(
        "RBLN needs FusedMoE.forward_oot to use RBLN MoE quant methods and "
        "RBLN data-parallel multicast/reduction behavior."
    ),
    priority=49,
    owner_module=__name__,
)(patched_fused_moe_forward_oot)


register_patch(
    target="vllm.model_executor.layers.fused_moe.layer.FusedMoE.naive_multicast",
    reason=(
        "RBLN needs FusedMoE naive_multicast to pad each DP input to the "
        "compiled max token bucket and gather all DP ranks."
    ),
    priority=49,
    owner_module=__name__,
)(patched_fused_moe_naive_multicast)


register_patch(
    target=(
        "vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod.apply"
    ),
    reason=(
        "RBLN routes unquantized MoE through the selected RBLN implementation "
        "based on VLLM_RBLN_MOE_USE_OPT_KERNEL and VLLM_RBLN_MOE_CUSTOM_KERNEL."
    ),
    priority=49,
    owner_module=__name__,
)(patched_unquantized_fused_moe_method_apply)
