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

from vllm_rbln.model_executor.models.qwen3_moe import (
    patched_qwen3_moe_sparse_moe_block_forward,
)
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward",
    reason=(
        "The RBLN path needs Qwen3MoE blocks to route experts through the "
        "patched router-callback path and combine the split shared/fused "
        "expert outputs before tensor-parallel reduction, because the "
        "upstream forward path precomputes router_logits and follows "
        "sequence-parallel and TP reduction rules that do not match the "
        "RBLN fused-MoE execution contract."
    ),
    owner_module=__name__,
)(patched_qwen3_moe_sparse_moe_block_forward)
