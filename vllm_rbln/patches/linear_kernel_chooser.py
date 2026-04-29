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

# This patch must be applied before creating model config because vLLM imports
# the mixed precision kernel chooser while building model config.

from vllm_rbln.model_executor.kernels.linear import patched_choose_mp_linear_kernel
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.kernels.linear.choose_mp_linear_kernel",
    reason=(
        "The RBLN path needs mixed precision linear layers to select the "
        "RBLN int8 unpacked kernel on RBLN devices while preserving upstream "
        "kernel selection for non-RBLN platforms."
    ),
    owner_module=__name__,
)(patched_choose_mp_linear_kernel)
