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
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock

from vllm_rbln.patches.patch_registry import register_patch

# NOTE(RBLN): This patch originated from
# https://github.com/RBLN-SW/vllm-rbln/commit/d6c5ec8960a6108e94698b71191e12e887c09184
# and its active router/shared-output behavior was updated in
# https://github.com/RBLN-SW/vllm-rbln/pull/367 and
# https://github.com/RBLN-SW/vllm-rbln/pull/511.


@register_patch(
    target="vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward",
    reason=(
        "The RBLN path needs Qwen3MoE blocks to route experts through the "
        "patched router-callback path and combine the split shared/fused "
        "expert outputs before tensor-parallel reduction, because the "
        "upstream forward path precomputes router_logits and follows "
        "sequence-parallel and TP reduction rules that do not match the "
        "RBLN fused-MoE execution contract."
    ),
)
def rbln_qwen3_moe_sparse_moe_block_forward(
    self: Qwen3MoeSparseMoeBlock,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # RBLN fused-MoE keeps the original token layout and expects router
    # computation to happen inside the patched expert path.
    shared_out, fused_out = self.experts(
        hidden_states=hidden_states,
        router=lambda x: self.gate(x)[0],
    )
    final_hidden_states = (
        shared_out + fused_out if shared_out is not None else fused_out
    )
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states
