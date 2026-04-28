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

import torch
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

import vllm_rbln.rbln_envs as envs
from vllm_rbln.patches.patch_registry import register_patch

# NOTE(RBLN): This patch originated from https://github.com/RBLN-SW/vllm-rbln/pull/81


@register_patch(
    target="vllm.model_executor.layers.logits_processor.LogitsProcessor._get_logits",
    reason=(
        "The RBLN TP logits path needs local LM-head execution to remain "
        "inside the compiled compute_logits graph while tensor-parallel "
        "gathering is handled separately when "
        "VLLM_RBLN_LOGITS_ALL_GATHER is disabled."
    ),
    condition=lambda: not envs.VLLM_RBLN_LOGITS_ALL_GATHER,
)
def logits_processor_get_logits(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    embedding_bias: torch.Tensor | None,
) -> torch.Tensor | None:
    logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
    return logits


@register_patch(
    target="vllm.model_executor.layers.logits_processor.LogitsProcessor._gather_logits",
    reason=(
        "The RBLN TP logits path needs tensor-parallel gathering and "
        "padded-vocabulary trimming to run as explicit post-processing "
        "outside the compiled compute_logits graph when "
        "VLLM_RBLN_LOGITS_ALL_GATHER is disabled."
    ),
    condition=lambda: not envs.VLLM_RBLN_LOGITS_ALL_GATHER,
)
def logits_processor_gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
    """gather/all-gather the logits tensor across model parallel group."""
    if self.use_all_gather:
        logits = tensor_model_parallel_all_gather(logits)
    else:
        logits = tensor_model_parallel_gather(logits)

    if logits is not None:
        logits = logits[..., : self.org_vocab_size]
    return logits
