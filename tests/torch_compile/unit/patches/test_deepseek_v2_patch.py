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
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2MoE,
)

from vllm_rbln.patches.models_deepseek_v2 import (
    rbln_deepseek_v2_attention_forward,
    rbln_deepseek_v2_moe_forward,
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


def _get_deepseek_v2_descriptors():
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.models_deepseek_v2"
    )


def test_deepseek_v2_patch_descriptors_are_registry_managed():
    descriptors = _get_deepseek_v2_descriptors()

    assert "vllm_rbln.models.deepseek_v2" not in get_legacy_patch_modules()
    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.models.deepseek_v2.DeepseekV2MoE.forward",
        "vllm.model_executor.models.deepseek_v2.DeepseekV2Attention.forward",
    }


def test_deepseek_v2_patch_descriptors_update_targets(monkeypatch):
    def original_moe_forward(*args, **kwargs):
        return args, kwargs

    def original_attention_forward(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(DeepseekV2MoE, "forward", original_moe_forward)
    monkeypatch.setattr(DeepseekV2Attention, "forward", original_attention_forward)

    apply_patch_descriptors(_get_deepseek_v2_descriptors())

    assert DeepseekV2MoE.forward is rbln_deepseek_v2_moe_forward
    assert DeepseekV2Attention.forward is rbln_deepseek_v2_attention_forward


def test_deepseek_v2_patch_default_verify_rejects_missing_assignments(monkeypatch):
    descriptors = _get_deepseek_v2_descriptors()

    def original_moe_forward(*args, **kwargs):
        return args, kwargs

    def original_attention_forward(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(DeepseekV2MoE, "forward", original_moe_forward)
    monkeypatch.setattr(DeepseekV2Attention, "forward", original_attention_forward)

    for descriptor in descriptors:
        with pytest.raises(RuntimeError, match="failed to patch target"):
            _verify_target_patch(descriptor)
