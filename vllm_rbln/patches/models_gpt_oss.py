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

from vllm_rbln.model_executor.models.gpt_oss import (
    patched_gpt_oss_load_weights_mxfp4,
    patched_gpt_oss_mlp_block_forward,
)
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.models.gpt_oss.GptOssModel._load_weights_mxfp4",
    reason=(
        "The RBLN path needs GPT-OSS MXFP4 weight loading to honor "
        "num_hidden_layers overrides while preserving the expert-weight "
        "slicing rules used by the RBLN MoE path, because the upstream "
        "loader assumes the full upstream layer set during weight traversal."
    ),
    owner_module=__name__,
)(patched_gpt_oss_load_weights_mxfp4)


register_patch(
    target="vllm.model_executor.models.gpt_oss.MLPBlock.forward",
    reason=(
        "The RBLN path needs GPT-OSS MoE blocks to route experts through the "
        "patched router callback and apply an explicit tensor-parallel "
        "all-reduce, because the upstream forward path uses "
        "sequence-parallel chunking and router_logits precomputation that do "
        "not match the RBLN fused-MoE execution contract."
    ),
    owner_module=__name__,
)(patched_gpt_oss_mlp_block_forward)
