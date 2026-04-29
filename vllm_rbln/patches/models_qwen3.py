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

from vllm_rbln.model_executor.models.qwen3 import patched_qwen3_for_causal_lm_init
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.models.qwen3.Qwen3ForCausalLM.__init__",
    reason=(
        "The RBLN path needs dense Qwen3 models to initialize a "
        "tensor-parallel ParallelLMHead even when word embeddings are tied, "
        "because token embeddings stay non-tensor-parallel while the LM "
        "head must remain tensor-parallel sharded."
    ),
    owner_module=__name__,
)(patched_qwen3_for_causal_lm_init)
