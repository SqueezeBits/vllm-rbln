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
from vllm.model_executor.models.minimax_m2 import MiniMaxM2MoE

from vllm_rbln.patches.models_minimax_m2 import rbln_minimax_m2_moe_forward
from vllm_rbln.patches.patch_registry import (
    _verify_target_patch,
    apply_patch_descriptors,
    get_registered_patch_descriptors,
)


@pytest.fixture(autouse=True)
def reset_patch_registry_state(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())


def _get_minimax_m2_descriptors():
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.models_minimax_m2"
    )


def test_minimax_m2_patch_descriptor_is_registry_managed():
    descriptors = _get_minimax_m2_descriptors()

    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.models.minimax_m2.MiniMaxM2MoE.forward",
    }


def test_minimax_m2_patch_descriptor_updates_target(monkeypatch):
    def original_forward(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(MiniMaxM2MoE, "forward", original_forward)

    apply_patch_descriptors(_get_minimax_m2_descriptors())

    assert MiniMaxM2MoE.forward is rbln_minimax_m2_moe_forward


def test_minimax_m2_patch_default_verify_rejects_missing_assignment(monkeypatch):
    descriptor = _get_minimax_m2_descriptors()[0]

    def original_forward(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(MiniMaxM2MoE, "forward", original_forward)

    with pytest.raises(RuntimeError, match="failed to patch target"):
        _verify_target_patch(descriptor)
