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

from vllm_rbln.model_executor.models.minimax_m2 import (
    patched_minimax_m2_moe_forward,
)
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.models.minimax_m2.MiniMaxM2MoE.forward",
    reason=(
        "The RBLN path needs MiniMaxM2 MoE blocks to route experts through "
        "the patched router callback with float32 gate evaluation and apply "
        "an explicit tensor-parallel all-reduce, because the upstream "
        "forward path flattens tokens, precomputes router_logits, and "
        "reshapes outputs in a way that does not match the RBLN fused-MoE "
        "execution contract."
    ),
    owner_module=__name__,
)(patched_minimax_m2_moe_forward)
