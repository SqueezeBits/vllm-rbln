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
import vllm.model_executor.layers.quantization.fp8 as upstream_fp8

from vllm_rbln.model_executor.layers.quantization.fp8 import (
    PatchedFp8LinearMethod,
    PatchedFp8MoEMethod,
)
from vllm_rbln.patches.patch_registry import (
    _verify_target_patch,
    apply_patch_descriptors,
    get_general_extension_modules,
    get_registered_patch_descriptors,
)


@pytest.fixture(autouse=True)
def reset_patch_registry_state(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())


def _get_fp8_patch_module():
    return import_module("vllm_rbln.patches.fp8")


def _get_fp8_descriptors():
    _get_fp8_patch_module()
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.fp8"
    )


def test_fp8_custom_op_module_is_registered_as_general_extension():
    custom_op_module = "vllm_rbln.model_executor.layers.quantization.fp8"

    assert custom_op_module in get_general_extension_modules()


def test_fp8_patch_descriptors_are_registry_managed():
    descriptors = _get_fp8_descriptors()

    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod",
        "vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod",
    }


def test_fp8_patch_descriptors_update_targets(monkeypatch):
    class OriginalFp8LinearMethod:
        pass

    class OriginalFp8MoEMethod:
        pass

    monkeypatch.setattr(
        upstream_fp8,
        "Fp8LinearMethod",
        OriginalFp8LinearMethod,
    )
    monkeypatch.setattr(
        upstream_fp8,
        "Fp8MoEMethod",
        OriginalFp8MoEMethod,
    )

    apply_patch_descriptors(_get_fp8_descriptors())

    assert upstream_fp8.Fp8LinearMethod is PatchedFp8LinearMethod
    assert upstream_fp8.Fp8MoEMethod is PatchedFp8MoEMethod


def test_fp8_patch_default_verify_rejects_missing_assignments(monkeypatch):
    descriptors = _get_fp8_descriptors()

    class OriginalFp8LinearMethod:
        pass

    class OriginalFp8MoEMethod:
        pass

    monkeypatch.setattr(
        upstream_fp8,
        "Fp8LinearMethod",
        OriginalFp8LinearMethod,
    )
    monkeypatch.setattr(
        upstream_fp8,
        "Fp8MoEMethod",
        OriginalFp8MoEMethod,
    )

    for descriptor in descriptors:
        with pytest.raises(RuntimeError, match="failed to patch target"):
            _verify_target_patch(descriptor)
