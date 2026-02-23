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
"""RBLN strict LMCache connector.

This connector keeps LMCacheConnectorV1 semantics but makes strict mode robust
on RBLN when LMCache cannot initialize a cache engine for the current device.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

import torch
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl as LMCacheConnectorLatestImpl,
)
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1 as UpstreamLMCacheConnectorV1,
    LMCacheKVEvents,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    import torch

    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)
_PATCH_LOCK = threading.Lock()
_PATCHED_LMCACHE_FOR_RBLN = False


def _is_rbln_platform() -> bool:
    return getattr(current_platform, "plugin_name", "") == "rbln"


def _build_local_disk_restore_defaults(
    metadata: "LMCacheMetadata | None",
) -> tuple[torch.Size, torch.dtype, MemoryFormat] | None:
    """Build fallback tensor metadata for disk entries restored on startup."""
    if metadata is None:
        return None

    num_layers = metadata.kv_shape[0]
    num_tokens = getattr(metadata, "chunk_size", metadata.kv_shape[2])
    if metadata.use_mla:
        hidden_dim = metadata.kv_shape[-1]
        return (
            torch.Size([1, num_layers, num_tokens, hidden_dim]),
            metadata.kv_dtype,
            MemoryFormat.KV_MLA_FMT,
        )

    hidden_dim = metadata.kv_shape[3] * metadata.kv_shape[4]
    return (
        torch.Size([2, num_layers, num_tokens, hidden_dim]),
        metadata.kv_dtype,
        MemoryFormat.KV_2LTD,
    )


def _maybe_restore_local_disk_key(local_disk_backend: Any, key: Any) -> bool:
    """Hydrate a missing LocalDiskBackend index entry when file exists on disk."""
    if key in local_disk_backend.dict:
        return True

    shape = getattr(local_disk_backend, "_rbln_local_disk_default_shape", None)
    dtype = getattr(local_disk_backend, "_rbln_local_disk_default_dtype", None)
    fmt = getattr(local_disk_backend, "_rbln_local_disk_default_fmt", None)
    if shape is None or dtype is None or fmt is None:
        return False

    path = local_disk_backend._key_to_path(key)
    if not os.path.exists(path):
        return False

    from lmcache.utils import DiskCacheMetadata

    size = os.path.getsize(path)
    local_disk_backend.dict[key] = DiskCacheMetadata(
        path,
        size,
        shape,
        dtype,
        None,
        fmt,
        0,
    )
    local_disk_backend.current_cache_size = (
        getattr(local_disk_backend, "current_cache_size", 0.0) + size
    )
    local_disk_backend.usage = getattr(local_disk_backend, "usage", 0.0) + size

    stats_monitor = getattr(local_disk_backend, "stats_monitor", None)
    if stats_monitor is not None:
        stats_monitor.update_local_storage_usage(local_disk_backend.usage)

    cache_policy = getattr(local_disk_backend, "cache_policy", None)
    if cache_policy is not None:
        try:
            cache_policy.update_on_put(key, local_disk_backend.dict)
        except TypeError:
            cache_policy.update_on_put(key)

    logger.info(
        "Restored LMCache local-disk index entry for chunk key on RBLN: %s",
        key,
    )
    return True


class RBLNHostGPUConnector(GPUConnectorInterface):
    """Host tensor connector used by LMCache engine on RBLN.

    LMCache's default vLLM connectors are CUDA/XPU specific. This connector
    performs the required page<->chunk copies with pure torch tensor indexing
    on host tensors.
    """

    def __init__(self, hidden_dim_size: int, num_layers: int, use_mla: bool = False):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_mla = use_mla
        self.kvcaches: list[torch.Tensor] | None = None

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: "torch.device | None" = None,
    ) -> "RBLNHostGPUConnector":
        del use_gpu, device
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_mla=metadata.use_mla,
        )

    @staticmethod
    def _get_slot_mapping_slice(
        slot_mapping: "torch.Tensor",
        start: int,
        end: int,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        slots = slot_mapping[start:end].to(dtype=torch.long, device="cpu")
        valid = slots >= 0
        return slots, valid

    @staticmethod
    def _read_non_mla_slots(kv_layer: "torch.Tensor", slots: "torch.Tensor") -> "torch.Tensor":
        if kv_layer.dim() == 5:
            # [2, num_blocks, block_size, num_heads, head_size]
            block_size = kv_layer.shape[2]
            block_idx = slots // block_size
            token_idx = slots % block_size
            picked = kv_layer[:, block_idx, token_idx, :, :]
            return picked.reshape(2, slots.numel(), -1)

        if kv_layer.dim() == 6:
            # RBLN paged format:
            # [2, num_blocks, num_heads, tp_size, block_size, head_size]
            block_size = kv_layer.shape[4]
            block_idx = slots // block_size
            token_idx = slots % block_size
            # Advanced indexing returns [num_slots, 2, num_heads, tp, head_size]
            # when both block and token indices are tensors.
            picked = kv_layer[:, block_idx, :, :, token_idx, :]
            picked = picked.permute(1, 0, 2, 3, 4)
            return picked.reshape(2, slots.numel(), -1)

        raise ValueError(f"Unsupported non-MLA kv_layer shape: {tuple(kv_layer.shape)}")

    @staticmethod
    def _write_non_mla_slots(
        kv_layer: "torch.Tensor",
        slots: "torch.Tensor",
        values: "torch.Tensor",
    ) -> None:
        if kv_layer.dim() == 5:
            block_size = kv_layer.shape[2]
            block_idx = slots // block_size
            token_idx = slots % block_size
            reshaped = values.reshape(
                2,
                slots.numel(),
                kv_layer.shape[3],
                kv_layer.shape[4],
            )
            kv_layer[:, block_idx, token_idx, :, :] = reshaped
            return

        if kv_layer.dim() == 6:
            block_size = kv_layer.shape[4]
            block_idx = slots // block_size
            token_idx = slots % block_size
            reshaped = values.reshape(
                2,
                slots.numel(),
                kv_layer.shape[2],
                kv_layer.shape[3],
                kv_layer.shape[5],
            )
            # Assignment target shape is [num_slots, 2, num_heads, tp, head_size].
            kv_layer[:, block_idx, :, :, token_idx, :] = reshaped.permute(
                1, 0, 2, 3, 4
            )
            return

        raise ValueError(f"Unsupported non-MLA kv_layer shape: {tuple(kv_layer.shape)}")

    @staticmethod
    def _flatten_mla_layer(kv_layer: "torch.Tensor") -> "torch.Tensor":
        # [num_pages, page_size, hidden] -> [num_slots, hidden]
        return kv_layer.reshape(kv_layer.shape[0] * kv_layer.shape[1], -1)

    def _validate_memory_format(self, memory_obj: MemoryObj) -> None:
        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format "
                    "for RBLNHostGPUConnector."
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format "
                    "for RBLNHostGPUConnector."
                )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None
        self._validate_memory_format(memory_obj)
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, "kvcaches should be initialized."
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slots, valid = self._get_slot_mapping_slice(kwargs["slot_mapping"], start, end)
        if not torch.any(valid):
            return

        if self.use_mla:
            src = memory_obj.tensor[0] if memory_obj.tensor.dim() == 4 else memory_obj.tensor
            for layer_idx, kv_layer in enumerate(self.kvcaches):
                flat = self._flatten_mla_layer(kv_layer)
                if torch.all(valid):
                    flat[slots, :] = src[layer_idx, :, :]
                else:
                    valid_slots = slots[valid]
                    flat[valid_slots, :] = src[layer_idx, valid, :]
            return

        src = memory_obj.tensor
        for layer_idx, kv_layer in enumerate(self.kvcaches):
            if torch.all(valid):
                self._write_non_mla_slots(kv_layer, slots, src[:, layer_idx, :, :])
            else:
                valid_slots = slots[valid]
                self._write_non_mla_slots(
                    kv_layer,
                    valid_slots,
                    src[:, layer_idx, valid, :],
                )

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, "kvcaches should be initialized."
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slots, valid = self._get_slot_mapping_slice(kwargs["slot_mapping"], start, end)

        if self.use_mla:
            dst = memory_obj.tensor[0] if memory_obj.tensor.dim() == 4 else memory_obj.tensor
            dst.zero_()
            if torch.any(valid):
                for layer_idx, kv_layer in enumerate(self.kvcaches):
                    flat = self._flatten_mla_layer(kv_layer)
                    dst[layer_idx, valid, :] = flat[slots[valid], :]
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            return

        dst = memory_obj.tensor
        dst.zero_()
        if torch.any(valid):
            for layer_idx, kv_layer in enumerate(self.kvcaches):
                dst[:, layer_idx, valid, :] = self._read_non_mla_slots(
                    kv_layer,
                    slots[valid],
                )
        memory_obj.metadata.fmt = MemoryFormat.KV_2LTD

    def batched_to_gpu(
        self,
        memory_objs: list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs,
    ):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def batched_from_gpu(
        self,
        memory_objs: list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs,
    ):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])


def _patch_lmcache_for_rbln() -> None:
    """Patch LMCache runtime hooks so vLLM LMCache can run on RBLN."""

    global _PATCHED_LMCACHE_FOR_RBLN
    if _PATCHED_LMCACHE_FOR_RBLN or not _is_rbln_platform():
        return

    with _PATCH_LOCK:
        if _PATCHED_LMCACHE_FOR_RBLN:
            return

        import lmcache.observability as lm_observability
        from lmcache.integration.vllm import utils as lm_utils
        from lmcache.v1 import gpu_connector as lm_gpu_connector
        from lmcache.v1 import manager as lm_manager
        from lmcache.v1.storage_backend import local_disk_backend as lm_local_disk_backend

        original_calc_local_rank = lm_utils.calculate_local_rank_and_world_size
        original_create_gpu_connector = lm_gpu_connector.CreateGPUConnector
        original_get_or_create_prom_logger = lm_observability.PrometheusLogger.GetOrCreate
        original_local_disk_init = lm_local_disk_backend.LocalDiskBackend.__init__
        original_local_disk_contains = lm_local_disk_backend.LocalDiskBackend.contains

        def _rbln_calculate_local_rank_and_world_size(
            vllm_config: VllmConfig,
        ) -> tuple[int, int]:
            if _is_rbln_platform():
                parallel_config = vllm_config.parallel_config
                return parallel_config.rank, parallel_config.world_size
            return original_calc_local_rank(vllm_config)

        def _rbln_create_gpu_connector(
            config: "LMCacheEngineConfig",
            metadata: "LMCacheMetadata",
            engine: str,
        ) -> GPUConnectorInterface:
            if _is_rbln_platform() and engine == "vllm":
                logger.info(
                    "Using RBLNHostGPUConnector for LMCache engine "
                    "(RBLN host tensor path)."
                )
                return RBLNHostGPUConnector.from_metadata(metadata)
            return original_create_gpu_connector(config, metadata, engine)

        def _rbln_get_or_create_prometheus_logger(metadata):
            if _is_rbln_platform():
                instance = lm_observability.PrometheusLogger._instance
                if instance is not None and instance.metadata != metadata:
                    logger.debug(
                        "Reusing existing LMCache PrometheusLogger on RBLN "
                        "despite metadata mismatch: existing=%s requested=%s",
                        instance.metadata,
                        metadata,
                    )
                    return instance
            return original_get_or_create_prom_logger(metadata)

        def _rbln_local_disk_init(
            self,
            config,
            loop,
            local_cpu_backend,
            dst_device="cuda",
            lmcache_worker=None,
            metadata=None,
        ):
            original_local_disk_init(
                self,
                config,
                loop,
                local_cpu_backend,
                dst_device=dst_device,
                lmcache_worker=lmcache_worker,
                metadata=metadata,
            )
            defaults = _build_local_disk_restore_defaults(metadata)
            if defaults is None:
                return
            (
                self._rbln_local_disk_default_shape,
                self._rbln_local_disk_default_dtype,
                self._rbln_local_disk_default_fmt,
            ) = defaults

        def _rbln_local_disk_contains(self, key, pin=False):
            with self.disk_lock:
                if key not in self.dict:
                    _maybe_restore_local_disk_key(self, key)

                if key not in self.dict:
                    return False

                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                return True

        lm_utils.calculate_local_rank_and_world_size = (
            _rbln_calculate_local_rank_and_world_size
        )
        lm_gpu_connector.CreateGPUConnector = _rbln_create_gpu_connector
        lm_manager.CreateGPUConnector = _rbln_create_gpu_connector
        lm_observability.PrometheusLogger.GetOrCreate = staticmethod(
            _rbln_get_or_create_prometheus_logger
        )
        lm_local_disk_backend.LocalDiskBackend.__init__ = _rbln_local_disk_init
        lm_local_disk_backend.LocalDiskBackend.contains = _rbln_local_disk_contains
        _PATCHED_LMCACHE_FOR_RBLN = True


class RBLNLMCacheConnectorV1Impl(LMCacheConnectorLatestImpl):
    """LMCache impl with degraded-mode guards for non-CUDA RBLN platforms."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ) -> None:
        _patch_lmcache_for_rbln()
        super().__init__(vllm_config, role, parent)
        self._warned_degraded_mode = False

    def _engine_ready(self, api_name: str) -> bool:
        if self.lmcache_engine is not None:
            return True

        if not self._warned_degraded_mode:
            logger.warning(
                "LMCache engine is unavailable in strict mode on this RBLN "
                "platform. Falling back to recompute/no-op behavior for "
                "LMCache APIs."
            )
            self._warned_degraded_mode = True

        logger.debug("Skipping LMCache API '%s' because engine is unavailable.", api_name)
        return False

    def _build_kv_layer_groups(self) -> None:
        """Disable LMCache layer-group inference for RBLN KV tensor shapes.

        LMCache currently expects vLLM GPU layouts when inferring grouped KV
        layer shapes. RBLN KV tensors have a different paged layout, so we
        intentionally keep groups empty and use metadata.kv_shape fallback.
        """
        if self.lmcache_engine is None:
            return
        self.lmcache_engine.metadata.kv_layer_groups_manager.kv_layer_groups = []

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if not self._engine_ready("start_load_kv"):
            return
        super().start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._engine_ready("wait_for_layer_load"):
            return
        super().wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        if not self._engine_ready("save_kv_layer"):
            return
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        if not self._engine_ready("wait_for_save"):
            return
        super().wait_for_save()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int | None:
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: "Request",
        num_external_tokens: int,
    ) -> None:
        super().update_state_after_alloc(request, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        return super().build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        # request_finished is safe without cache engine and can still return
        # transfer params for disagg response paths.
        return super().request_finished(request, block_ids)


class RBLNLMCacheConnectorV1(UpstreamLMCacheConnectorV1):
    """vLLM LMCache connector wrapper using RBLN strict impl."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        KVConnectorBase_V1.__init__(
            self,
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        logger.info("Initializing RBLN strict LMCache connector")
        self._lmcache_engine = RBLNLMCacheConnectorV1Impl(vllm_config, role, self)
        self._kv_cache_events: LMCacheKVEvents | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Forward KV cache registration to strict LMCache engine.

        Upstream LMCacheConnectorV1 currently keeps register_kv_caches as a
        no-op. RBLN strict mode requires explicit registration so LMCache
        avoids legacy forward-context KV cache discovery.
        """
        self._lmcache_engine.register_kv_caches(kv_caches)
