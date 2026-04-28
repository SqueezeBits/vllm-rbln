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

# NOTE(RBLN): This patch originated from
# https://github.com/RBLN-SW/vllm-rbln/commit/eff874a while syncing the RBLN
# attention layer with upstream in https://github.com/RBLN-SW/vllm-rbln/pull/509,
# and was later updated through https://github.com/RBLN-SW/vllm-rbln/commit/fbcaf43
# for KV cache input deduplication in https://github.com/RBLN-SW/vllm-rbln/pull/524.

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_rbln.model_executor.layers.attention.attention import (
    _rbln_attention_init,
    _rbln_get_kv_cache_spec,
    _rbln_unified_attention,
    _rbln_unified_attention_with_output,
)
from vllm_rbln.patches.patch_registry import register_patch


@register_patch(
    target="vllm.model_executor.layers.attention.attention.unified_attention",
    reason=(
        "RBLN needs unified_attention to resolve KV cache from attention "
        "metadata or deduplicated KV-cache base tensors instead of the "
        "attention layer's embedded KV cache."
    ),
)
@maybe_transfer_kv_layer
def rbln_unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return _rbln_unified_attention(query, key, value, layer_name)


@register_patch(
    target=(
        "vllm.model_executor.layers.attention.attention.unified_attention_with_output"
    ),
    reason=(
        "RBLN needs unified_attention_with_output to preserve the KV-cache "
        "dummy dependency and resolve KV cache from attention metadata or "
        "deduplicated KV-cache base tensors."
    ),
)
@maybe_transfer_kv_layer
def rbln_unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return _rbln_unified_attention_with_output(
        query,
        key,
        value,
        output,
        layer_name,
        output_scale,
        output_block_scale,
        kv_cache_dummy_dep,
    )


@register_patch(
    target="vllm.model_executor.layers.attention.attention.Attention.__init__",
    reason=(
        "RBLN needs Attention initialization to record a pipeline-adjusted "
        "layer index so external KV-cache bindings can resolve the matching "
        "per-layer cache tensor."
    ),
)
def rbln_attention_init(self, *args, **kwargs) -> None:
    return _rbln_attention_init(self, *args, **kwargs)


@register_patch(
    target=(
        "vllm.model_executor.layers.attention.attention.Attention.get_kv_cache_spec"
    ),
    reason=(
        "RBLN needs Attention KV-cache specs to use RBLN sliding-window "
        "metadata while preserving upstream full-attention spec fields."
    ),
)
def rbln_get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
    return _rbln_get_kv_cache_spec(self, vllm_config)
