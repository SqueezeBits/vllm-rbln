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

from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE

from vllm_rbln.model_executor.layers.fused_moe.shared_fused_moe import (
    patched_deleted_shared_fused_moe_forward,
    patched_shared_fused_moe_forward_oot,
    patched_shared_fused_moe_init,
)
from vllm_rbln.patches.patch_registry import register_patch


def _verify_shared_fused_moe_forward_deleted() -> None:
    if "forward" in SharedFusedMoE.__dict__:
        delattr(SharedFusedMoE, "forward")

    if "forward" in SharedFusedMoE.__dict__:
        msg = "failed to delete patch target: SharedFusedMoE.forward"
        raise RuntimeError(msg)


register_patch(
    target=(
        "vllm.model_executor.layers.fused_moe.shared_fused_moe.SharedFusedMoE.__init__"
    ),
    reason=(
        "RBLN needs SharedFusedMoE initialization to reuse FusedMoE setup while "
        "disabling overlapped shared expert execution, which is not supported."
    ),
    owner_module=__name__,
)(patched_shared_fused_moe_init)


register_patch(
    target=(
        "vllm.model_executor.layers.fused_moe.shared_fused_moe."
        "SharedFusedMoE.forward_oot"
    ),
    reason=(
        "RBLN needs SharedFusedMoE to route CustomOp.forward through "
        "forward_oot and apply tensor-parallel reduction to shared experts."
    ),
    owner_module=__name__,
)(patched_shared_fused_moe_forward_oot)


register_patch(
    target=(
        "vllm.model_executor.layers.fused_moe.shared_fused_moe.SharedFusedMoE.forward"
    ),
    reason=(
        "RBLN removes upstream SharedFusedMoE.forward so CustomOp.forward "
        "dispatches to the patched forward_oot implementation."
    ),
    verify=_verify_shared_fused_moe_forward_deleted,
    owner_module=__name__,
)(patched_deleted_shared_fused_moe_forward)
