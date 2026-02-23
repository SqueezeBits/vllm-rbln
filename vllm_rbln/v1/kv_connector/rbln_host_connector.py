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
"""Host-mediated KV connector for RBLN disaggregation fallback.

This connector reuses vLLM's ExampleConnector scheduler-side behavior but
loads KV tensors into the current layer device instead of hard-coding CUDA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import safetensors
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    ExampleConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext


class RBLNHostConnector(ExampleConnector):
    """Slow but robust host-storage KV transfer path for RBLN."""

    def _load_kv_cache(
        self, filename: str, target_device: torch.device
    ) -> torch.Tensor:
        kv_cache = safetensors.torch.load_file(filename)["kv_cache"]
        if kv_cache.device != target_device:
            kv_cache = kv_cache.to(device=target_device)
        return kv_cache

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Load KV cache from host storage and inject into paged memory."""
        del kwargs
        attn_metadata = forward_context.attn_metadata

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1
                )
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1
                )
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ExampleConnectorMetadata)

        if metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the connector metadata is None"
            )
            return

        if attn_metadata is None:
            logger.warning("In connector.start_load_kv, but the attn_metadata is None")
            return

        for request in metadata.requests:
            if request.is_store:
                continue
            logger.info(
                "Inject KV cache of %d tokens to the paged memory",
                len(request.slot_mapping),
            )
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue

                kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                filename = self._generate_filename_debug(
                    layer_name, request.token_ids, request.mm_hashes
                )
                kv_cache = self._load_kv_cache(filename, kv_cache_layer.device)
                inject_kv_into_layer(kv_cache_layer, kv_cache, request.slot_mapping)
