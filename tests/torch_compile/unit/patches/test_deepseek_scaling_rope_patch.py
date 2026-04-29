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
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)

from vllm_rbln.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    patched_deepseek_scaling_rotary_embedding_forward,
)
from vllm_rbln.patches.patch_registry import (
    _verify_target_patch,
    apply_patch_descriptors,
    get_registered_patch_descriptors,
)


@pytest.fixture(autouse=True)
def reset_patch_registry_state(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())


def _get_deepseek_scaling_rope_descriptors():
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.deepseek_scaling_rope"
    )


def test_deepseek_scaling_rope_patch_descriptor_is_registry_managed():
    descriptors = _get_deepseek_scaling_rope_descriptors()

    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope.DeepseekScalingRotaryEmbedding.forward",  # noqa: E501
    }


def test_deepseek_scaling_rope_patch_descriptor_updates_target(monkeypatch):
    def original_forward(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(DeepseekScalingRotaryEmbedding, "forward", original_forward)

    apply_patch_descriptors(_get_deepseek_scaling_rope_descriptors())

    assert (
        DeepseekScalingRotaryEmbedding.forward
        is patched_deepseek_scaling_rotary_embedding_forward
    )


def test_deepseek_scaling_rope_patch_default_verify_rejects_missing_assignment(
    monkeypatch,
):
    descriptor = _get_deepseek_scaling_rope_descriptors()[0]

    def original_forward(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(DeepseekScalingRotaryEmbedding, "forward", original_forward)

    with pytest.raises(RuntimeError, match="failed to patch target"):
        _verify_target_patch(descriptor)
