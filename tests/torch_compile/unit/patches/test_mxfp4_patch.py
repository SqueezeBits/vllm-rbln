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
import vllm.model_executor.layers.quantization.mxfp4 as upstream_mxfp4

from vllm_rbln.patches.patch_registry import (
    _verify_target_patch,
    apply_patch_descriptors,
    get_general_extension_modules,
    get_legacy_patch_modules,
    get_registered_patch_descriptors,
)


@pytest.fixture(autouse=True)
def reset_patch_registry_state(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())


def _get_mxfp4_patch_module():
    return import_module("vllm_rbln.patches.mxfp4")


def _get_mxfp4_descriptors():
    _get_mxfp4_patch_module()
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.mxfp4"
    )


def test_mxfp4_custom_op_module_is_registered_as_general_extension():
    custom_op_module = "vllm_rbln.model_executor.layers.quantization.mxfp4"

    assert custom_op_module in get_general_extension_modules()
    assert custom_op_module not in get_legacy_patch_modules()


def test_mxfp4_patch_descriptor_is_registry_managed():
    descriptors = _get_mxfp4_descriptors()

    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod",
    }


def test_mxfp4_patch_descriptor_updates_target(monkeypatch):
    mxfp4_patch = _get_mxfp4_patch_module()

    class OriginalMxfp4MoEMethod:
        pass

    monkeypatch.setattr(
        upstream_mxfp4,
        "Mxfp4MoEMethod",
        OriginalMxfp4MoEMethod,
    )

    apply_patch_descriptors(_get_mxfp4_descriptors())

    assert upstream_mxfp4.Mxfp4MoEMethod is mxfp4_patch.Mxfp4MoEMethod


def test_mxfp4_patch_default_verify_rejects_missing_assignment(monkeypatch):
    descriptor = _get_mxfp4_descriptors()[0]

    class OriginalMxfp4MoEMethod:
        pass

    monkeypatch.setattr(
        upstream_mxfp4,
        "Mxfp4MoEMethod",
        OriginalMxfp4MoEMethod,
    )

    with pytest.raises(RuntimeError, match="failed to patch target"):
        _verify_target_patch(descriptor)
