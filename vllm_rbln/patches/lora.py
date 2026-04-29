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

from vllm_rbln.lora.layers import patched_vocab_parallel_embedding_with_lora_forward
from vllm_rbln.lora.layers.base_linear import patched_base_linear_apply
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.lora.layers.base_linear.BaseLinearLayerWithLoRA.apply",
    reason=(
        "Adapt LoRA linear composition to the RBLN vLLM-model execution path, "
        "where the effective input/output tensor shapes differ from the "
        "upstream vLLM path."
    ),
    owner_module=__name__,
)(patched_base_linear_apply)


register_patch(
    target="vllm.lora.layers.VocabParallelEmbeddingWithLoRA.forward",
    reason=(
        "Adapt LoRA embedding composition to the RBLN vLLM-model execution "
        "path, where prefill/decode tensor shapes and Punica metadata layout "
        "differ from the upstream vLLM path."
    ),
    owner_module=__name__,
)(patched_vocab_parallel_embedding_with_lora_forward)
