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

from vllm_rbln.model_executor.layers.quantization.fp8 import (
    PatchedFp8LinearMethod,
    PatchedFp8MoEMethod,
)
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod",
    reason=(
        "RBLN needs a custom FP8 linear method because the RBLN path uses "
        "W8A16 block FP8 dequantization and a BF16 fallback linear operation "
        "instead of upstream GPU FP8 kernels."
    ),
    owner_module=__name__,
)(PatchedFp8LinearMethod)


register_patch(
    target="vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod",
    reason=(
        "RBLN needs a custom FP8 MoE method because the RBLN path lowers "
        "FP8 MoE through rbln_custom_ops::custom_moe_swiglu_group_dequantize "
        "with static-shape-friendly expert routing."
    ),
    owner_module=__name__,
)(PatchedFp8MoEMethod)
