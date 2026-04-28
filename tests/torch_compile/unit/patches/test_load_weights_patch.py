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
from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.llama4 import Llama4Model
from vllm.model_executor.models.minimax_m2 import MiniMaxM2Model
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen2_moe import Qwen2MoeModel
from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel

from vllm_rbln.patches.load_weights import (
    load_deepseek_v2_weights,
    load_llama4_weights,
    load_llama_weights,
    load_minimax_m2_weights,
    load_qwen2_weights,
    load_qwen2moe_weights,
    load_qwen3moe_weights,
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


def _get_load_weights_descriptors():
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.load_weights"
    )


def test_load_weights_patch_descriptors_are_registry_managed():
    descriptors = _get_load_weights_descriptors()

    assert "vllm_rbln.model_executor.model_loader.weight_loader" not in (
        get_legacy_patch_modules()
    )
    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.models.llama.LlamaModel.load_weights",
        "vllm.model_executor.models.llama4.Llama4Model.load_weights",
        "vllm.model_executor.models.qwen2.Qwen2Model.load_weights",
        "vllm.model_executor.models.qwen2_moe.Qwen2MoeModel.load_weights",
        "vllm.model_executor.models.qwen3_moe.Qwen3MoeModel.load_weights",
        "vllm.model_executor.models.deepseek_v2.DeepseekV2ForCausalLM.load_weights",
        "vllm.model_executor.models.minimax_m2.MiniMaxM2Model.load_weights",
    }


def test_load_weights_patch_descriptors_update_targets(monkeypatch):
    def original_load_weights(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(LlamaModel, "load_weights", original_load_weights)
    monkeypatch.setattr(Llama4Model, "load_weights", original_load_weights)
    monkeypatch.setattr(Qwen2Model, "load_weights", original_load_weights)
    monkeypatch.setattr(Qwen2MoeModel, "load_weights", original_load_weights)
    monkeypatch.setattr(Qwen3MoeModel, "load_weights", original_load_weights)
    monkeypatch.setattr(DeepseekV2ForCausalLM, "load_weights", original_load_weights)
    monkeypatch.setattr(MiniMaxM2Model, "load_weights", original_load_weights)

    apply_patch_descriptors(_get_load_weights_descriptors())

    assert LlamaModel.load_weights is load_llama_weights
    assert Llama4Model.load_weights is load_llama4_weights
    assert Qwen2Model.load_weights is load_qwen2_weights
    assert Qwen2MoeModel.load_weights is load_qwen2moe_weights
    assert Qwen3MoeModel.load_weights is load_qwen3moe_weights
    assert DeepseekV2ForCausalLM.load_weights is load_deepseek_v2_weights
    assert MiniMaxM2Model.load_weights is load_minimax_m2_weights


def test_load_weights_patch_default_verify_rejects_missing_assignments(monkeypatch):
    descriptors = _get_load_weights_descriptors()

    def original_load_weights(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(LlamaModel, "load_weights", original_load_weights)
    monkeypatch.setattr(Llama4Model, "load_weights", original_load_weights)
    monkeypatch.setattr(Qwen2Model, "load_weights", original_load_weights)
    monkeypatch.setattr(Qwen2MoeModel, "load_weights", original_load_weights)
    monkeypatch.setattr(Qwen3MoeModel, "load_weights", original_load_weights)
    monkeypatch.setattr(DeepseekV2ForCausalLM, "load_weights", original_load_weights)
    monkeypatch.setattr(MiniMaxM2Model, "load_weights", original_load_weights)

    for descriptor in descriptors:
        with pytest.raises(RuntimeError, match="failed to patch target"):
            _verify_target_patch(descriptor)
