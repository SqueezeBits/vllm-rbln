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

import pytest
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)

from vllm_rbln.model_executor.layers.fused_moe.layer import (
    patched_fused_moe_forward_oot,
    patched_fused_moe_init,
    patched_fused_moe_naive_multicast,
    patched_unquantized_fused_moe_method_apply,
)
from vllm_rbln.patches.patch_registry import (
    apply_patch_descriptors,
    get_general_extension_modules,
    get_registered_patch_descriptors,
)


@pytest.fixture(autouse=True)
def reset_patch_registry_state(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())


@pytest.fixture(autouse=True)
def restore_fused_moe_methods():
    original_init = FusedMoE.__init__
    original_forward_oot = FusedMoE.forward_oot
    had_naive_multicast = hasattr(FusedMoE, "naive_multicast")
    original_naive_multicast = getattr(FusedMoE, "naive_multicast", None)
    original_apply = UnquantizedFusedMoEMethod.apply

    yield

    FusedMoE.__init__ = original_init
    FusedMoE.forward_oot = original_forward_oot
    if had_naive_multicast:
        FusedMoE.naive_multicast = original_naive_multicast
    elif hasattr(FusedMoE, "naive_multicast"):
        del FusedMoE.naive_multicast
    UnquantizedFusedMoEMethod.apply = original_apply


def _get_fused_moe_patch_module():
    return import_module("vllm_rbln.patches.fused_moe")


def _get_fused_moe_descriptors():
    _get_fused_moe_patch_module()
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.fused_moe"
    )


def test_fused_moe_patch_descriptors_are_registry_managed():
    descriptors = _get_fused_moe_descriptors()

    assert "vllm_rbln.model_executor.layers.fused_moe.custom_ops" in (
        get_general_extension_modules()
    )
    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.layers.fused_moe.layer.FusedMoE.__init__",
        "vllm.model_executor.layers.fused_moe.layer.FusedMoE.forward_oot",
        "vllm.model_executor.layers.fused_moe.layer.FusedMoE.naive_multicast",
        ("vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod.apply"),
    }


def test_fused_moe_patch_descriptors_precede_shared_fused_moe_descriptors():
    import_module("vllm_rbln.patches.shared_fused_moe")

    descriptors = get_registered_patch_descriptors()
    fused_descriptors = [
        descriptor
        for descriptor in descriptors
        if descriptor.owner_module == "vllm_rbln.patches.fused_moe"
    ]
    shared_descriptors = [
        descriptor
        for descriptor in descriptors
        if descriptor.owner_module == "vllm_rbln.patches.shared_fused_moe"
    ]
    fused_indices = [
        index
        for index, descriptor in enumerate(descriptors)
        if descriptor.owner_module == "vllm_rbln.patches.fused_moe"
    ]
    shared_indices = [
        index
        for index, descriptor in enumerate(descriptors)
        if descriptor.owner_module == "vllm_rbln.patches.shared_fused_moe"
    ]

    assert fused_indices
    assert shared_indices
    assert {descriptor.priority for descriptor in fused_descriptors} == {49}
    assert {descriptor.priority for descriptor in shared_descriptors} == {50}
    assert max(fused_indices) < min(shared_indices)


def test_fused_moe_patch_descriptors_update_targets(monkeypatch):
    def original_init(self, *args, **kwargs):
        return None

    def original_forward_oot(self, *args, **kwargs):
        return args, kwargs

    def original_naive_multicast(self, *args, **kwargs):
        return args, kwargs

    def original_apply(self, *args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(FusedMoE, "__init__", original_init)
    monkeypatch.setattr(FusedMoE, "forward_oot", original_forward_oot)
    monkeypatch.setattr(FusedMoE, "naive_multicast", original_naive_multicast)
    monkeypatch.setattr(UnquantizedFusedMoEMethod, "apply", original_apply)

    apply_patch_descriptors(_get_fused_moe_descriptors())

    assert FusedMoE.__init__ is patched_fused_moe_init
    assert FusedMoE.forward_oot is patched_fused_moe_forward_oot
    assert FusedMoE.naive_multicast is patched_fused_moe_naive_multicast
    assert UnquantizedFusedMoEMethod.apply is patched_unquantized_fused_moe_method_apply
