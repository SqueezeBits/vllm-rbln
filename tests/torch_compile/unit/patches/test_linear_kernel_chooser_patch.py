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
import vllm.model_executor.kernels.linear as linear

from vllm_rbln.patches.patch_registry import (
    _verify_target_patch,
    apply_patch_descriptors,
    get_registered_patch_descriptors,
)


@pytest.fixture(autouse=True)
def reset_patch_registry_state(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())


def _get_linear_kernel_chooser_module():
    return import_module("vllm_rbln.patches.linear_kernel_chooser")


def _get_linear_kernel_chooser_descriptors():
    _get_linear_kernel_chooser_module()
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.linear_kernel_chooser"
    )


def test_linear_kernel_chooser_patch_descriptor_is_registry_managed():
    descriptors = _get_linear_kernel_chooser_descriptors()

    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.kernels.linear.choose_mp_linear_kernel",
    }


def test_linear_kernel_chooser_patch_descriptor_updates_target(monkeypatch):
    linear_kernel_chooser = _get_linear_kernel_chooser_module()

    def original_choose_mp_linear_kernel(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(
        linear,
        "choose_mp_linear_kernel",
        original_choose_mp_linear_kernel,
    )

    apply_patch_descriptors(_get_linear_kernel_chooser_descriptors())

    assert linear.choose_mp_linear_kernel is (
        linear_kernel_chooser.choose_mp_linear_kernel_rbln
    )


def test_linear_kernel_chooser_patch_default_verify_rejects_missing_assignment(
    monkeypatch,
):
    descriptor = _get_linear_kernel_chooser_descriptors()[0]

    def original_choose_mp_linear_kernel(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(
        linear,
        "choose_mp_linear_kernel",
        original_choose_mp_linear_kernel,
    )

    with pytest.raises(RuntimeError, match="failed to patch target"):
        _verify_target_patch(descriptor)
