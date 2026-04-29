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

from vllm_rbln.model_executor.models.deepseek_v2 import (
    patched_deepseek_v2_attention_forward,
    patched_deepseek_v2_moe_forward,
)
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.models.deepseek_v2.DeepseekV2MoE.forward",
    reason=(
        "The RBLN path needs DeepseekV2 MoE blocks to route experts through "
        "the patched router-callback path while preserving the model-specific "
        "shared-expert and FP16 scaling behavior, because the upstream "
        "forward path precomputes router_logits and follows sequence-parallel "
        "and TP reduction rules that do not match the RBLN fused-MoE "
        "execution contract."
    ),
    owner_module=__name__,
)(patched_deepseek_v2_moe_forward)


register_patch(
    target="vllm.model_executor.models.deepseek_v2.DeepseekV2Attention.forward",
    reason=(
        "The RBLN path needs a dedicated DeepseekV2 MLA attention path that "
        "reassembles q/k/v tensors around the patched rotary outputs and "
        "head-dimension alignment, because the upstream forward path does "
        "not preserve the tensor shapes expected by RBLN execution."
    ),
    owner_module=__name__,
)(patched_deepseek_v2_attention_forward)
