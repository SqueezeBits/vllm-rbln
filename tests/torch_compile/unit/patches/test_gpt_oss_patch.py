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
from vllm.model_executor.models.gpt_oss import GptOssModel, MLPBlock

from vllm_rbln.patches.models_gpt_oss import (
    rbln_gpt_oss_load_weights_mxfp4,
    rbln_gpt_oss_mlp_block_forward,
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


def _get_gpt_oss_descriptors():
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.models_gpt_oss"
    )


def test_gpt_oss_patch_descriptors_are_registry_managed():
    descriptors = _get_gpt_oss_descriptors()

    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.models.gpt_oss.GptOssModel._load_weights_mxfp4",
        "vllm.model_executor.models.gpt_oss.MLPBlock.forward",
    }


def test_gpt_oss_patch_descriptors_update_targets(monkeypatch):
    def original_load_weights(*args, **kwargs):
        return args, kwargs

    def original_forward(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(GptOssModel, "_load_weights_mxfp4", original_load_weights)
    monkeypatch.setattr(MLPBlock, "forward", original_forward)

    apply_patch_descriptors(_get_gpt_oss_descriptors())

    assert GptOssModel._load_weights_mxfp4 is rbln_gpt_oss_load_weights_mxfp4
    assert MLPBlock.forward is rbln_gpt_oss_mlp_block_forward


def test_gpt_oss_patch_default_verify_rejects_missing_assignments(monkeypatch):
    descriptors = _get_gpt_oss_descriptors()

    def original_load_weights(*args, **kwargs):
        return args, kwargs

    def original_forward(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(GptOssModel, "_load_weights_mxfp4", original_load_weights)
    monkeypatch.setattr(MLPBlock, "forward", original_forward)

    for descriptor in descriptors:
        with pytest.raises(RuntimeError, match="failed to patch target"):
            _verify_target_patch(descriptor)
