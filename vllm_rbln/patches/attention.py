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

from vllm_rbln.model_executor.layers.attention.attention import (
    patched_attention_init,
    patched_get_kv_cache_spec,
    patched_unified_attention,
    patched_unified_attention_with_output,
)
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.layers.attention.attention.unified_attention",
    reason=(
        "RBLN needs unified_attention to resolve KV cache from attention "
        "metadata or deduplicated KV-cache base tensors instead of the "
        "attention layer's embedded KV cache."
    ),
    owner_module=__name__,
)(patched_unified_attention)


register_patch(
    target=(
        "vllm.model_executor.layers.attention.attention.unified_attention_with_output"
    ),
    reason=(
        "RBLN needs unified_attention_with_output to preserve the KV-cache "
        "dummy dependency and resolve KV cache from attention metadata or "
        "deduplicated KV-cache base tensors."
    ),
    owner_module=__name__,
)(patched_unified_attention_with_output)


register_patch(
    target="vllm.model_executor.layers.attention.attention.Attention.__init__",
    reason=(
        "RBLN needs Attention initialization to record a pipeline-adjusted "
        "layer index so external KV-cache bindings can resolve the matching "
        "per-layer cache tensor."
    ),
    owner_module=__name__,
)(patched_attention_init)


register_patch(
    target=(
        "vllm.model_executor.layers.attention.attention.Attention.get_kv_cache_spec"
    ),
    reason=(
        "RBLN needs Attention KV-cache specs to use RBLN sliding-window "
        "metadata while preserving upstream full-attention spec fields."
    ),
    owner_module=__name__,
)(patched_get_kv_cache_spec)
