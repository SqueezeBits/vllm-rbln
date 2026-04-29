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
from vllm.lora.layers import VocabParallelEmbeddingWithLoRA
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

from vllm_rbln.patches.lora import (
    base_linear_patched_apply,
    vocab_parallel_embedding_patched_forward,
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


def _get_lora_descriptors():
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.lora"
    )


def test_lora_patch_descriptors_are_registry_managed():
    descriptors = _get_lora_descriptors()

    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.lora.layers.base_linear.BaseLinearLayerWithLoRA.apply",
        "vllm.lora.layers.VocabParallelEmbeddingWithLoRA.forward",
    }


def test_lora_patch_descriptors_update_targets(monkeypatch):
    def original_base_linear_apply(*args, **kwargs):
        return args, kwargs

    def original_vocab_parallel_embedding_forward(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(BaseLinearLayerWithLoRA, "apply", original_base_linear_apply)
    monkeypatch.setattr(
        VocabParallelEmbeddingWithLoRA,
        "forward",
        original_vocab_parallel_embedding_forward,
    )

    apply_patch_descriptors(_get_lora_descriptors())

    assert BaseLinearLayerWithLoRA.apply is base_linear_patched_apply
    assert (
        VocabParallelEmbeddingWithLoRA.forward
        is vocab_parallel_embedding_patched_forward
    )


def test_lora_patch_default_verify_rejects_missing_assignment(monkeypatch):
    descriptor = next(
        descriptor
        for descriptor in _get_lora_descriptors()
        if descriptor.target.endswith("BaseLinearLayerWithLoRA.apply")
    )

    def original_base_linear_apply(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(BaseLinearLayerWithLoRA, "apply", original_base_linear_apply)

    with pytest.raises(RuntimeError, match="failed to patch target"):
        _verify_target_patch(descriptor)
