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
import vllm.forward_context as upstream_forward_context

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


def _get_forward_context_patch_module():
    return import_module("vllm_rbln.patches.forward_context")


def _get_forward_context_descriptors():
    _get_forward_context_patch_module()
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.forward_context"
    )


def test_forward_context_patch_descriptor_is_registry_managed():
    descriptors = _get_forward_context_descriptors()

    assert "vllm_rbln.forward_context" not in get_legacy_patch_modules()
    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.forward_context.set_forward_context",
    }


def test_forward_context_patch_descriptor_updates_target(monkeypatch):
    forward_context_patch = _get_forward_context_patch_module()

    def original_set_forward_context(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(
        upstream_forward_context,
        "set_forward_context",
        original_set_forward_context,
    )

    apply_patch_descriptors(_get_forward_context_descriptors())

    assert upstream_forward_context.set_forward_context is (
        forward_context_patch.rbln_set_forward_context
    )


def test_forward_context_patch_default_verify_rejects_missing_assignment(
    monkeypatch,
):
    descriptor = _get_forward_context_descriptors()[0]

    def original_set_forward_context(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(
        upstream_forward_context,
        "set_forward_context",
        original_set_forward_context,
    )

    with pytest.raises(RuntimeError, match="failed to patch target"):
        _verify_target_patch(descriptor)
