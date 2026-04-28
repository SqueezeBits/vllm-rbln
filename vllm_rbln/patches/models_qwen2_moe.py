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
from vllm.model_executor.models.qwen2_moe import Qwen2MoeSparseMoeBlock

from vllm_rbln.patches.patch_registry import register_patch

# NOTE(RBLN): This patch originated from https://github.com/RBLN-SW/vllm-rbln/pull/145
# and its active v0.12 forward variant was updated in
# https://github.com/RBLN-SW/vllm-rbln/commit/fd0f28fd60042a93ef4deff0a5f99cc28ffb0643.


@register_patch(
    target="vllm.model_executor.models.qwen2_moe.Qwen2MoeSparseMoeBlock.forward",
    reason=(
        "The RBLN path needs Qwen2MoE blocks to keep the original "
        "hidden-state layout and route experts through the patched router "
        "callback path, because the upstream forward path flattens tokens "
        "and precomputes router_logits in a way that does not match the "
        "RBLN fused-MoE execution contract."
    ),
)
def rbln_qwen2_moe_sparse_moe_block_forward(
    self: Qwen2MoeSparseMoeBlock,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # NOTE(RBLN): fused-MoE keeps the original token layout; do not flatten here.
    final_hidden_states = self.experts(
        hidden_states=hidden_states,
        router=lambda x: self.gate(x)[0],
    )
    if self.shared_expert is not None:
        final_hidden_states = final_hidden_states[0] + final_hidden_states[1]
    if self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states
        )
    return final_hidden_states
