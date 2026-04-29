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

from vllm_rbln.model_executor.models.deepseek_v2 import (
    patched_deepseek_v2_load_weights,
)
from vllm_rbln.model_executor.models.llama import patched_llama_load_weights
from vllm_rbln.model_executor.models.llama4 import patched_llama4_load_weights
from vllm_rbln.model_executor.models.minimax_m2 import (
    patched_minimax_m2_load_weights,
)
from vllm_rbln.model_executor.models.qwen2 import patched_qwen2_load_weights
from vllm_rbln.model_executor.models.qwen2_moe import (
    patched_qwen2_moe_load_weights,
)
from vllm_rbln.model_executor.models.qwen3_moe import (
    patched_qwen3_moe_load_weights,
)
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.models.llama.LlamaModel.load_weights",
    reason=(
        "The RBLN path needs Llama weight loading to honor num_hidden_layers "
        "overrides while preserving the model-specific stacked-projection "
        "and kv-scale loading rules."
    ),
    owner_module=__name__,
)(patched_llama_load_weights)


register_patch(
    target="vllm.model_executor.models.llama4.Llama4Model.load_weights",
    reason=(
        "The RBLN path needs Llama4 weight loading to honor num_hidden_layers "
        "overrides while preserving the model-specific fused-expert and "
        "kv-scale loading rules."
    ),
    owner_module=__name__,
)(patched_llama4_load_weights)


register_patch(
    target="vllm.model_executor.models.qwen2.Qwen2Model.load_weights",
    reason=(
        "The RBLN path needs Qwen2 weight loading to honor num_hidden_layers "
        "overrides while preserving the model-specific stacked-projection "
        "and kv-scale loading rules."
    ),
    owner_module=__name__,
)(patched_qwen2_load_weights)


register_patch(
    target="vllm.model_executor.models.qwen2_moe.Qwen2MoeModel.load_weights",
    reason=(
        "The RBLN path needs Qwen2MoE weight loading to honor "
        "num_hidden_layers overrides while preserving the model-specific "
        "stacked-projection, expert-weight, GGUF remapping, and kv-scale "
        "loading rules."
    ),
    owner_module=__name__,
)(patched_qwen2_moe_load_weights)


register_patch(
    target="vllm.model_executor.models.qwen3_moe.Qwen3MoeModel.load_weights",
    reason=(
        "The RBLN path needs Qwen3MoE weight loading to honor "
        "num_hidden_layers overrides while preserving the model-specific "
        "stacked-projection, expert-weight, and kv-scale loading rules."
    ),
    owner_module=__name__,
)(patched_qwen3_moe_load_weights)


register_patch(
    target="vllm.model_executor.models.deepseek_v2.DeepseekV2ForCausalLM.load_weights",
    reason=(
        "The RBLN path needs DeepseekV2 weight loading to honor "
        "num_hidden_layers overrides while skipping spec-decode layers and "
        "preserving the model-specific expert-weight and kv-scale loading "
        "rules."
    ),
    owner_module=__name__,
)(patched_deepseek_v2_load_weights)


register_patch(
    target="vllm.model_executor.models.minimax_m2.MiniMaxM2Model.load_weights",
    reason=(
        "The RBLN path needs MiniMaxM2 weight loading to honor "
        "num_hidden_layers overrides while skipping spec-decode layers and "
        "preserving the model-specific expert-weight and kv-scale loading "
        "rules."
    ),
    owner_module=__name__,
)(patched_minimax_m2_load_weights)
