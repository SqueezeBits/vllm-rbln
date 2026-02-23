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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("lmcache.integration.vllm.vllm_v1_adapter")

from lmcache.utils import CacheEngineKey
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata
from lmcache.v1.memory_management import MemoryFormat

from vllm_rbln.v1.kv_connector.rbln_lmcache_connector import (
    RBLNHostGPUConnector,
    RBLNLMCacheConnectorV1,
    RBLNLMCacheConnectorV1Impl,
    _build_local_disk_restore_defaults,
    _maybe_restore_local_disk_key,
    _patch_lmcache_for_rbln,
)


def _build_impl_without_engine() -> RBLNLMCacheConnectorV1Impl:
    impl = RBLNLMCacheConnectorV1Impl.__new__(RBLNLMCacheConnectorV1Impl)
    impl._warned_degraded_mode = False
    impl._manager = SimpleNamespace(lmcache_engine=None)
    return impl


def test_engine_ready_returns_false_when_engine_missing():
    impl = _build_impl_without_engine()
    assert impl._engine_ready("test-api") is False
    assert impl._warned_degraded_mode is True


def test_missing_engine_methods_are_safe_noops():
    impl = _build_impl_without_engine()

    # Methods below should be no-ops and must not raise.
    impl.start_load_kv(SimpleNamespace())
    impl.wait_for_layer_load("layer.0")
    impl.save_kv_layer("layer.0", None, None)
    impl.wait_for_save()


def test_scheduler_side_methods_delegate_to_super_apis():
    impl = _build_impl_without_engine()
    request = SimpleNamespace()
    scheduler_output = SimpleNamespace()

    with (
        patch(
            "lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl.get_num_new_matched_tokens",
            return_value=7,
        ) as get_matched,
        patch(
            "lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl.update_state_after_alloc",
            return_value=None,
        ) as update_alloc,
        patch(
            "lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl.build_connector_meta",
            return_value=LMCacheConnectorMetadata(),
        ) as build_meta,
    ):
        assert impl.get_num_new_matched_tokens(request, 0) == 7
        impl.update_state_after_alloc(request, 0)
        metadata = impl.build_connector_meta(scheduler_output)

    get_matched.assert_called_once_with(request, 0)
    update_alloc.assert_called_once_with(request, 0)
    build_meta.assert_called_once_with(scheduler_output)
    assert isinstance(metadata, LMCacheConnectorMetadata)


def test_wrapper_register_kv_caches_delegates_to_strict_impl():
    class _Engine:
        def __init__(self):
            self.called = False
            self.kv_caches = None

        def register_kv_caches(self, kv_caches):
            self.called = True
            self.kv_caches = kv_caches

    connector = RBLNLMCacheConnectorV1.__new__(RBLNLMCacheConnectorV1)
    connector._lmcache_engine = _Engine()

    kv_caches = {"layer.0": torch.zeros(1)}
    connector.register_kv_caches(kv_caches)

    assert connector._lmcache_engine.called is True
    assert connector._lmcache_engine.kv_caches is kv_caches


def test_patch_lmcache_for_rbln_suppresses_prometheus_metadata_mismatch_error(
    monkeypatch: pytest.MonkeyPatch,
):
    import lmcache.observability as lm_observability

    from vllm_rbln.v1.kv_connector import rbln_lmcache_connector as module_under_test

    monkeypatch.setattr(module_under_test, "_PATCHED_LMCACHE_FOR_RBLN", False)
    monkeypatch.setattr(module_under_test, "_is_rbln_platform", lambda: True)
    _patch_lmcache_for_rbln()

    logged_errors: list[tuple[tuple, dict]] = []

    def _capture_error(*args, **kwargs):
        logged_errors.append((args, kwargs))

    monkeypatch.setattr(lm_observability.logger, "error", _capture_error)

    previous_instance = lm_observability.PrometheusLogger._instance
    try:
        lm_observability.PrometheusLogger._instance = SimpleNamespace(
            metadata="worker-metadata"
        )
        reused = lm_observability.PrometheusLogger.GetOrCreate("scheduler-metadata")
    finally:
        lm_observability.PrometheusLogger._instance = previous_instance

    assert reused.metadata == "worker-metadata"
    assert logged_errors == []


def _make_non_mla_kvcaches() -> list[torch.Tensor]:
    num_layers = 2
    num_pages = 4
    page_size = 2
    hidden_dim = 3
    caches: list[torch.Tensor] = []
    for layer_idx in range(num_layers):
        base = layer_idx * 1000
        layer = torch.arange(
            base,
            base + 2 * num_pages * page_size * hidden_dim,
            dtype=torch.float32,
        ).reshape(2, num_pages, page_size, 1, hidden_dim)
        caches.append(layer)
    return caches


def _make_non_mla_kvcaches_6d() -> list[torch.Tensor]:
    num_layers = 2
    num_blocks = 4
    num_heads = 2
    tp_size = 2
    block_size = 2
    head_size = 3
    caches: list[torch.Tensor] = []
    for layer_idx in range(num_layers):
        base = layer_idx * 1000
        layer = torch.arange(
            base,
            base + 2 * num_blocks * num_heads * tp_size * block_size * head_size,
            dtype=torch.float32,
        ).reshape(2, num_blocks, num_heads, tp_size, block_size, head_size)
        caches.append(layer)
    return caches


def _flatten_kv_layer(kv_layer: torch.Tensor) -> torch.Tensor:
    return kv_layer.reshape(2, kv_layer.shape[1] * kv_layer.shape[2], -1)


def _flatten_kv_layer_6d(kv_layer: torch.Tensor) -> torch.Tensor:
    return kv_layer.permute(0, 1, 4, 2, 3, 5).reshape(
        2, kv_layer.shape[1] * kv_layer.shape[4], -1
    )


def test_host_gpu_connector_batched_from_gpu_reads_slots():
    kvcaches = _make_non_mla_kvcaches()
    connector = RBLNHostGPUConnector(hidden_dim_size=3, num_layers=2, use_mla=False)
    slot_mapping = torch.tensor([0, 3, 5], dtype=torch.long)
    memory_obj = SimpleNamespace(
        tensor=torch.empty((2, 2, 3, 3), dtype=torch.float32),
        metadata=SimpleNamespace(fmt=MemoryFormat.UNDEFINED),
    )

    connector.batched_from_gpu(
        memory_objs=[memory_obj],
        starts=[0],
        ends=[3],
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
    )

    expected = torch.stack(
        [_flatten_kv_layer(kv_layer)[:, slot_mapping, :] for kv_layer in kvcaches],
        dim=1,
    )
    assert torch.equal(memory_obj.tensor, expected)
    assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD


def test_host_gpu_connector_batched_to_gpu_writes_slots():
    source_caches = _make_non_mla_kvcaches()
    connector = RBLNHostGPUConnector(hidden_dim_size=3, num_layers=2, use_mla=False)
    slot_mapping = torch.tensor([0, 3, 5], dtype=torch.long)
    memory_tensor = torch.stack(
        [_flatten_kv_layer(kv_layer)[:, slot_mapping, :] for kv_layer in source_caches],
        dim=1,
    )
    memory_obj = SimpleNamespace(
        tensor=memory_tensor.clone(),
        metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
    )

    target_caches = [torch.zeros_like(kv_layer) for kv_layer in source_caches]

    connector.batched_to_gpu(
        memory_objs=[memory_obj],
        starts=[0],
        ends=[3],
        kvcaches=target_caches,
        slot_mapping=slot_mapping,
    )

    for layer_idx, kv_layer in enumerate(target_caches):
        flat = _flatten_kv_layer(kv_layer)
        expected = memory_tensor[:, layer_idx, :, :]
        assert torch.equal(flat[:, slot_mapping, :], expected)


def test_host_gpu_connector_batched_from_gpu_reads_slots_for_rbln_6d_layout():
    kvcaches = _make_non_mla_kvcaches_6d()
    connector = RBLNHostGPUConnector(hidden_dim_size=12, num_layers=2, use_mla=False)
    slot_mapping = torch.tensor([0, 3, 5], dtype=torch.long)
    memory_obj = SimpleNamespace(
        tensor=torch.empty((2, 2, 3, 12), dtype=torch.float32),
        metadata=SimpleNamespace(fmt=MemoryFormat.UNDEFINED),
    )

    connector.batched_from_gpu(
        memory_objs=[memory_obj],
        starts=[0],
        ends=[3],
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
    )

    expected = torch.stack(
        [_flatten_kv_layer_6d(kv_layer)[:, slot_mapping, :] for kv_layer in kvcaches],
        dim=1,
    )
    assert torch.equal(memory_obj.tensor, expected)
    assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD


def test_host_gpu_connector_batched_to_gpu_writes_slots_for_rbln_6d_layout():
    source_caches = _make_non_mla_kvcaches_6d()
    connector = RBLNHostGPUConnector(hidden_dim_size=12, num_layers=2, use_mla=False)
    slot_mapping = torch.tensor([0, 3, 5], dtype=torch.long)
    memory_tensor = torch.stack(
        [_flatten_kv_layer_6d(kv_layer)[:, slot_mapping, :] for kv_layer in source_caches],
        dim=1,
    )
    memory_obj = SimpleNamespace(
        tensor=memory_tensor.clone(),
        metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
    )
    target_caches = [torch.zeros_like(kv_layer) for kv_layer in source_caches]

    connector.batched_to_gpu(
        memory_objs=[memory_obj],
        starts=[0],
        ends=[3],
        kvcaches=target_caches,
        slot_mapping=slot_mapping,
    )

    for layer_idx, kv_layer in enumerate(target_caches):
        flat = _flatten_kv_layer_6d(kv_layer)
        expected = memory_tensor[:, layer_idx, :, :]
        assert torch.equal(flat[:, slot_mapping, :], expected)


def test_build_local_disk_restore_defaults_non_mla():
    metadata = SimpleNamespace(
        use_mla=False,
        kv_shape=(16, 2, 256, 8, 64),
        kv_dtype=torch.bfloat16,
    )

    defaults = _build_local_disk_restore_defaults(metadata)

    assert defaults is not None
    shape, dtype, fmt = defaults
    assert shape == torch.Size([2, 16, 256, 512])
    assert dtype == torch.bfloat16
    assert fmt == MemoryFormat.KV_2LTD


def test_maybe_restore_local_disk_key_hydrates_missing_key_from_file(tmp_path):
    key = CacheEngineKey(
        model_name="meta-llama/Llama-3.2-1B",
        world_size=1,
        worker_id=0,
        chunk_hash=0x1234,
        dtype=torch.bfloat16,
        request_configs=None,
    )
    path = tmp_path / f"{key.to_string().replace('/', '-')}.pt"
    path.write_bytes(b"0123456789")

    stats_calls: list[float] = []
    policy_calls: list[CacheEngineKey] = []
    backend = SimpleNamespace(
        dict={},
        _key_to_path=lambda _: str(path),
        _rbln_local_disk_default_shape=torch.Size([2, 16, 256, 512]),
        _rbln_local_disk_default_dtype=torch.bfloat16,
        _rbln_local_disk_default_fmt=MemoryFormat.KV_2LTD,
        current_cache_size=0.0,
        usage=0.0,
        stats_monitor=SimpleNamespace(
            update_local_storage_usage=lambda v: stats_calls.append(v)
        ),
        cache_policy=SimpleNamespace(
            update_on_put=lambda k, _mapping: policy_calls.append(k)
        ),
    )

    restored = _maybe_restore_local_disk_key(backend, key)

    assert restored is True
    assert key in backend.dict
    assert backend.dict[key].path == str(path)
    assert backend.dict[key].size == 10
    assert backend.current_cache_size == 10
    assert backend.usage == 10
    assert stats_calls == [10]
    assert policy_calls == [key]
