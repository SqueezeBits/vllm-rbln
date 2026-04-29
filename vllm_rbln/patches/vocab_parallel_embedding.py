# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from vllm_rbln.model_executor.layers.vocab_parallel_embedding import (
    patched_parallel_lm_head_tie_weights,
    patched_vocab_parallel_embedding_forward,
    patched_vocab_parallel_embedding_init,
)
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding.__init__",
    reason=(
        "The RBLN path needs token embeddings to stay replicated while "
        "keeping ParallelLMHead tensor-parallel sharded, because the "
        "upstream implementation assumes every vocab-parallel tensor "
        "follows the same TP-sharded padded-vocabulary layout."
    ),
    owner_module=__name__,
)(patched_vocab_parallel_embedding_init)


register_patch(
    target="vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding.forward",
    reason=(
        "The RBLN path needs regular token embedding lookups to bypass the "
        "upstream forward_native sharded-reduction path, while preserving "
        "masking and all-reduce only for layers that remain "
        "tensor-parallel sharded."
    ),
    owner_module=__name__,
)(patched_vocab_parallel_embedding_forward)


register_patch(
    target="vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead.tie_weights",
    reason=(
        "The RBLN path needs to avoid tying a tensor-parallel-sharded LM "
        "head directly to replicated token embeddings, while preserving "
        "the single-rank and GGUF cases."
    ),
    owner_module=__name__,
)(patched_parallel_lm_head_tie_weights)
