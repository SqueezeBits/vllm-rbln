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

from vllm_rbln.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    patched_deepseek_scaling_rotary_embedding_forward,
)
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope.DeepseekScalingRotaryEmbedding.forward",
    reason=(
        "The RBLN path needs a dedicated DeepSeek scaling RoPE execution "
        "path that materializes cos/sin caches with explicit "
        "index_select-based layout handling, because the upstream "
        "forward_native path is not a stable torch.compile target for RBLN."
    ),
    owner_module=__name__,
)(patched_deepseek_scaling_rotary_embedding_forward)
