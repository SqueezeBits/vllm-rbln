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

import pytest
from vllm.model_executor.layers.logits_processor import LogitsProcessor

import vllm_rbln.rbln_envs as envs
from vllm_rbln.patches.logits_processor import (
    logits_processor_gather_logits,
    logits_processor_get_logits,
)
from vllm_rbln.patches.patch_registry import (
    _verify_target_patch,
    apply_patch_descriptors,
    get_legacy_patch_modules,
    get_registered_patch_descriptors,
)


@pytest.fixture(autouse=True)
def reset_patch_registry_state(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())


def _get_logits_descriptors():
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.logits_processor"
    )


def test_logits_processor_patch_descriptors_are_registry_managed():
    descriptors = _get_logits_descriptors()

    assert "vllm_rbln.model_executor.layers.logits_processor" not in (
        get_legacy_patch_modules()
    )
    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.layers.logits_processor.LogitsProcessor._get_logits",
        "vllm.model_executor.layers.logits_processor.LogitsProcessor._gather_logits",
    }


def test_logits_processor_patch_descriptors_update_targets(monkeypatch):
    def original_get_logits(*args, **kwargs):
        return args, kwargs

    def original_gather_logits(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(envs, "VLLM_RBLN_LOGITS_ALL_GATHER", False)
    monkeypatch.setattr(LogitsProcessor, "_get_logits", original_get_logits)
    monkeypatch.setattr(LogitsProcessor, "_gather_logits", original_gather_logits)

    apply_patch_descriptors(_get_logits_descriptors())

    assert LogitsProcessor._get_logits is logits_processor_get_logits
    assert LogitsProcessor._gather_logits is logits_processor_gather_logits


def test_logits_processor_patch_is_disabled_when_all_gather_enabled(monkeypatch):
    def original_get_logits(*args, **kwargs):
        return args, kwargs

    def original_gather_logits(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(envs, "VLLM_RBLN_LOGITS_ALL_GATHER", True)
    monkeypatch.setattr(LogitsProcessor, "_get_logits", original_get_logits)
    monkeypatch.setattr(LogitsProcessor, "_gather_logits", original_gather_logits)

    apply_patch_descriptors(_get_logits_descriptors())

    assert LogitsProcessor._get_logits is original_get_logits
    assert LogitsProcessor._gather_logits is original_gather_logits


def test_logits_processor_patch_default_verify_rejects_missing_assignment(monkeypatch):
    descriptor = next(
        descriptor
        for descriptor in _get_logits_descriptors()
        if descriptor.target.endswith("LogitsProcessor._get_logits")
    )

    def original_get_logits(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(envs, "VLLM_RBLN_LOGITS_ALL_GATHER", False)
    monkeypatch.setattr(LogitsProcessor, "_get_logits", original_get_logits)

    with pytest.raises(RuntimeError, match="failed to patch target"):
        _verify_target_patch(descriptor)
