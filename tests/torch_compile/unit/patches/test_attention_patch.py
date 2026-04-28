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

from importlib import import_module
from inspect import signature

import pytest
import vllm.model_executor.layers.attention.attention as upstream_attention
from vllm.model_executor.layers.attention.attention import Attention

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


def _get_attention_patch_module():
    return import_module("vllm_rbln.patches.attention")


def _get_attention_descriptors():
    _get_attention_patch_module()
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.attention"
    )


def test_attention_patch_descriptors_are_registry_managed():
    descriptors = _get_attention_descriptors()

    assert (
        "vllm_rbln.model_executor.layers.attention.attention"
        not in get_legacy_patch_modules()
    )
    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.layers.attention.attention.unified_attention",
        "vllm.model_executor.layers.attention.attention.unified_attention_with_output",
        "vllm.model_executor.layers.attention.attention.Attention.__init__",
        "vllm.model_executor.layers.attention.attention.Attention.get_kv_cache_spec",
    }


def test_attention_patch_descriptors_update_targets(monkeypatch):
    attention_patch = _get_attention_patch_module()

    def original_unified_attention(*args, **kwargs):
        return args, kwargs

    def original_unified_attention_with_output(*args, **kwargs):
        return args, kwargs

    def original_init(self, *args, **kwargs):
        return None

    def original_get_kv_cache_spec(self, vllm_config):
        return vllm_config

    monkeypatch.setattr(
        upstream_attention,
        "unified_attention",
        original_unified_attention,
    )
    monkeypatch.setattr(
        upstream_attention,
        "unified_attention_with_output",
        original_unified_attention_with_output,
    )
    monkeypatch.setattr(Attention, "__init__", original_init)
    monkeypatch.setattr(Attention, "get_kv_cache_spec", original_get_kv_cache_spec)

    apply_patch_descriptors(_get_attention_descriptors())

    assert (
        upstream_attention.unified_attention is attention_patch.rbln_unified_attention
    )
    assert upstream_attention.unified_attention_with_output is (
        attention_patch.rbln_unified_attention_with_output
    )
    assert Attention.__init__ is attention_patch.rbln_attention_init
    assert Attention.get_kv_cache_spec is attention_patch.rbln_get_kv_cache_spec


def test_attention_transfer_wrappers_preserve_layer_name_signature():
    attention_patch = _get_attention_patch_module()

    assert "layer_name" in signature(attention_patch.rbln_unified_attention).parameters
    assert (
        "layer_name"
        in signature(attention_patch.rbln_unified_attention_with_output).parameters
    )


def test_attention_patch_default_verify_rejects_missing_assignments(monkeypatch):
    descriptors = _get_attention_descriptors()

    def original_unified_attention(*args, **kwargs):
        return args, kwargs

    def original_unified_attention_with_output(*args, **kwargs):
        return args, kwargs

    def original_init(self, *args, **kwargs):
        return None

    def original_get_kv_cache_spec(self, vllm_config):
        return vllm_config

    monkeypatch.setattr(
        upstream_attention,
        "unified_attention",
        original_unified_attention,
    )
    monkeypatch.setattr(
        upstream_attention,
        "unified_attention_with_output",
        original_unified_attention_with_output,
    )
    monkeypatch.setattr(Attention, "__init__", original_init)
    monkeypatch.setattr(Attention, "get_kv_cache_spec", original_get_kv_cache_spec)

    for descriptor in descriptors:
        with pytest.raises(RuntimeError, match="failed to patch target"):
            _verify_target_patch(descriptor)
