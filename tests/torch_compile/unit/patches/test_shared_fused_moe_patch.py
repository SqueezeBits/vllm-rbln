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
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE

from vllm_rbln.model_executor.layers.fused_moe.shared_fused_moe import (
    patched_shared_fused_moe_forward_oot,
    patched_shared_fused_moe_init,
)
from vllm_rbln.patches.patch_registry import (
    apply_patch_descriptors,
    get_registered_patch_descriptors,
)


@pytest.fixture(autouse=True)
def reset_patch_registry_state(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())


@pytest.fixture(autouse=True)
def restore_shared_fused_moe_methods():
    original_init = SharedFusedMoE.__init__
    had_forward_oot = hasattr(SharedFusedMoE, "forward_oot")
    original_forward_oot = getattr(SharedFusedMoE, "forward_oot", None)
    had_forward = hasattr(SharedFusedMoE, "forward")
    original_forward = getattr(SharedFusedMoE, "forward", None)

    yield

    SharedFusedMoE.__init__ = original_init
    if had_forward_oot:
        SharedFusedMoE.forward_oot = original_forward_oot
    elif hasattr(SharedFusedMoE, "forward_oot"):
        del SharedFusedMoE.forward_oot

    if had_forward:
        SharedFusedMoE.forward = original_forward
    elif hasattr(SharedFusedMoE, "forward"):
        del SharedFusedMoE.forward


def _get_shared_fused_moe_patch_module():
    return import_module("vllm_rbln.patches.shared_fused_moe")


def _get_shared_fused_moe_descriptors():
    _get_shared_fused_moe_patch_module()
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.shared_fused_moe"
    )


def test_shared_fused_moe_patch_descriptors_are_registry_managed():
    descriptors = _get_shared_fused_moe_descriptors()

    assert {descriptor.target for descriptor in descriptors} == {
        (
            "vllm.model_executor.layers.fused_moe.shared_fused_moe."
            "SharedFusedMoE.__init__"
        ),
        (
            "vllm.model_executor.layers.fused_moe.shared_fused_moe."
            "SharedFusedMoE.forward_oot"
        ),
        (
            "vllm.model_executor.layers.fused_moe.shared_fused_moe."
            "SharedFusedMoE.forward"
        ),
    }


def test_shared_fused_moe_patch_descriptors_update_targets():
    def original_init(self, *args, **kwargs):
        return None

    def original_forward_oot(self, *args, **kwargs):
        return args, kwargs

    def original_forward(self, *args, **kwargs):
        return args, kwargs

    SharedFusedMoE.__init__ = original_init
    SharedFusedMoE.forward_oot = original_forward_oot
    SharedFusedMoE.forward = original_forward

    apply_patch_descriptors(_get_shared_fused_moe_descriptors())

    assert SharedFusedMoE.__init__ is patched_shared_fused_moe_init
    assert SharedFusedMoE.forward_oot is patched_shared_fused_moe_forward_oot
    assert "forward" not in SharedFusedMoE.__dict__
