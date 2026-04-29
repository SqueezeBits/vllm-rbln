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
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from vllm_rbln.model_executor.layers.rotary_embedding.base import (
    patched_rope_forward_oot,
    patched_rotary_embedding_init,
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


def _get_rotary_embedding_descriptors():
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.rotary_embedding"
    )


def test_rotary_embedding_patch_descriptors_are_registry_managed():
    descriptors = _get_rotary_embedding_descriptors()

    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.__init__",
        "vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.forward_oot",
    }


def test_rotary_embedding_patch_descriptors_update_targets(monkeypatch):
    def original_init(*args, **kwargs):
        return args, kwargs

    def original_forward_oot(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(RotaryEmbedding, "__init__", original_init)
    monkeypatch.setattr(RotaryEmbedding, "forward_oot", original_forward_oot)

    apply_patch_descriptors(_get_rotary_embedding_descriptors())

    assert RotaryEmbedding.__init__ is patched_rotary_embedding_init
    assert RotaryEmbedding.forward_oot is patched_rope_forward_oot


def test_rotary_embedding_patch_default_verify_rejects_missing_assignment(monkeypatch):
    descriptor = next(
        descriptor
        for descriptor in _get_rotary_embedding_descriptors()
        if descriptor.target.endswith("RotaryEmbedding.__init__")
    )

    def original_init(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(RotaryEmbedding, "__init__", original_init)

    with pytest.raises(RuntimeError, match="failed to patch target"):
        _verify_target_patch(descriptor)
