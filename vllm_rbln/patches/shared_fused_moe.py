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

# NOTE(RBLN): This patch originated from
# https://github.com/RBLN-SW/vllm-rbln/commit/a614262 while updating GPT-OSS
# shared MoE behavior in https://github.com/RBLN-SW/vllm-rbln/pull/293, and was
# later updated through https://github.com/RBLN-SW/vllm-rbln/commit/191b133 and
# https://github.com/RBLN-SW/vllm-rbln/commit/cc949b8.

from typing import NoReturn

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE

from vllm_rbln.patches.patch_registry import register_patch


def _verify_shared_fused_moe_forward_deleted() -> None:
    if "forward" in SharedFusedMoE.__dict__:
        delattr(SharedFusedMoE, "forward")

    if "forward" in SharedFusedMoE.__dict__:
        msg = "failed to delete patch target: SharedFusedMoE.forward"
        raise RuntimeError(msg)


@register_patch(
    target=(
        "vllm.model_executor.layers.fused_moe.shared_fused_moe.SharedFusedMoE.__init__"
    ),
    reason=(
        "RBLN needs SharedFusedMoE initialization to reuse FusedMoE setup while "
        "disabling overlapped shared expert execution, which is not supported."
    ),
)
def rbln_shared_fused_moe_init(
    self: SharedFusedMoE,
    shared_experts: torch.nn.Module | None,
    gate: torch.nn.Module | None = None,
    use_overlapped: bool = True,
    **kwargs,
) -> None:
    del use_overlapped
    FusedMoE.__init__(self, **kwargs)
    self._shared_experts = shared_experts

    # RBLN does not support the upstream overlapped shared expert path.
    self.use_overlapped = False

    self._gate = gate


@register_patch(
    target=(
        "vllm.model_executor.layers.fused_moe.shared_fused_moe."
        "SharedFusedMoE.forward_oot"
    ),
    reason=(
        "RBLN needs SharedFusedMoE to route CustomOp.forward through "
        "forward_oot and apply tensor-parallel reduction to shared experts."
    ),
)
def rbln_shared_fused_moe_forward_oot(
    self: SharedFusedMoE,
    hidden_states: torch.Tensor,
    router: torch.nn.Module,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if not self.use_overlapped:
        if self._shared_experts is not None:
            shared_out = self._shared_experts(hidden_states)

            # The shared MLP is created with reduce_results=False, so reduce
            # here only when upstream would require the shared expert outputs.
            if (
                self.reduce_results
                and get_tensor_model_parallel_world_size() > 1
                and self.must_reduce_shared_expert_outputs()
            ):
                shared_out = tensor_model_parallel_all_reduce(shared_out)
        else:
            shared_out = None

        fused_out = FusedMoE.forward_oot(
            self,
            hidden_states=hidden_states,
            router=router,
        )
    else:
        shared_out, fused_out = FusedMoE.forward(
            self,
            hidden_states=hidden_states,
            router=router,
        )
        if (
            shared_out is not None
            and self.reduce_results
            and get_tensor_model_parallel_world_size() > 1
            and self.must_reduce_shared_expert_outputs()
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)
    return shared_out, fused_out


@register_patch(
    target=(
        "vllm.model_executor.layers.fused_moe.shared_fused_moe.SharedFusedMoE.forward"
    ),
    reason=(
        "RBLN removes upstream SharedFusedMoE.forward so CustomOp.forward "
        "dispatches to the patched forward_oot implementation."
    ),
    verify=_verify_shared_fused_moe_forward_deleted,
)
def rbln_deleted_shared_fused_moe_forward(*args, **kwargs) -> NoReturn:
    msg = "SharedFusedMoE.forward should be deleted after patch application"
    raise RuntimeError(msg)
