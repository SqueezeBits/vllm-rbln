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

from vllm_rbln.model_executor.layers.vocab_parallel_embedding import (
    patched_parallel_lm_head_tie_weights,
    patched_vocab_parallel_embedding_forward,
    patched_vocab_parallel_embedding_init,
)
from vllm_rbln.patches import register_patch

# NOTE(RBLN): Introduced in https://github.com/RBLN-SW/vllm-rbln/pull/81

register_patch(
    target="vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding.__init__",
    reason=(
        "RBLN tensor parallelism uses a split embedding layout: regular token "
        "embeddings stay replicated, while ParallelLMHead remains vocab-sharded "
        "per TP rank. Override initialization so plain VocabParallelEmbedding "
        "uses tp_size=1 and only ParallelLMHead uses the real TP rank/world size."
    ),
    owner_module=__name__,
)(patched_vocab_parallel_embedding_init)


register_patch(
    target="vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding.forward",
    reason=(
        "Plain token embeddings are replicated in the RBLN TP path, so their "
        "lookup result should be returned directly without upstream masking and "
        "TP all-reduce. Keep the sharded forward path only for vocab-parallel "
        "layers whose tp_size remains greater than one."
    ),
    owner_module=__name__,
)(patched_vocab_parallel_embedding_forward)


register_patch(
    target="vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead.tie_weights",
    reason=(
        "In RBLN TP, tied word embeddings cannot share one Parameter because "
        "embed_tokens is replicated while ParallelLMHead is vocab-sharded. "
        "Only alias weights for single-rank execution; TP loading is handled "
        "by replaying the tied embedding weight through lm_head.weight_loader."
    ),
    owner_module=__name__,
)(patched_parallel_lm_head_tie_weights)
