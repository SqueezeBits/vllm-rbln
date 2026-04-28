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
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_rbln.patches.patch_registry import (
    _verify_target_patch,
    apply_patch_descriptors,
    get_legacy_patch_modules,
    get_registered_patch_descriptors,
)
from vllm_rbln.patches.vocab_parallel_embedding import (
    parallel_lm_head_tie_weights,
    vocab_parallel_embedding_forward,
    vocab_parallel_embedding_init,
)


@pytest.fixture(autouse=True)
def reset_patch_registry_state(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())


def _get_vocab_parallel_embedding_descriptors():
    return tuple(
        descriptor
        for descriptor in get_registered_patch_descriptors()
        if descriptor.owner_module == "vllm_rbln.patches.vocab_parallel_embedding"
    )


def test_vocab_parallel_embedding_patch_descriptors_are_registry_managed():
    descriptors = _get_vocab_parallel_embedding_descriptors()

    assert "vllm_rbln.model_executor.layers.vocab_parallel_embedding" not in (
        get_legacy_patch_modules()
    )
    assert {descriptor.target for descriptor in descriptors} == {
        "vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding.__init__",  # noqa: E501
        "vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding.forward",  # noqa: E501
        "vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead.tie_weights",  # noqa: E501
    }


def test_vocab_parallel_embedding_patch_descriptors_update_targets(monkeypatch):
    def original_init(*args, **kwargs):
        return args, kwargs

    def original_forward(*args, **kwargs):
        return args, kwargs

    def original_tie_weights(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(VocabParallelEmbedding, "__init__", original_init)
    monkeypatch.setattr(VocabParallelEmbedding, "forward", original_forward)
    monkeypatch.setattr(ParallelLMHead, "tie_weights", original_tie_weights)

    apply_patch_descriptors(_get_vocab_parallel_embedding_descriptors())

    assert VocabParallelEmbedding.__init__ is vocab_parallel_embedding_init
    assert VocabParallelEmbedding.forward is vocab_parallel_embedding_forward
    assert ParallelLMHead.tie_weights is parallel_lm_head_tie_weights


def test_vocab_parallel_embedding_patch_default_verify_rejects_missing_assignment(
    monkeypatch,
):
    descriptor = next(
        descriptor
        for descriptor in _get_vocab_parallel_embedding_descriptors()
        if descriptor.target.endswith("VocabParallelEmbedding.__init__")
    )

    def original_init(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(VocabParallelEmbedding, "__init__", original_init)

    with pytest.raises(RuntimeError, match="failed to patch target"):
        _verify_target_patch(descriptor)
