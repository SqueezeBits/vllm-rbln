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

from vllm_rbln.model_executor.layers.quantization.mxfp4 import PatchedMxfp4MoEMethod
from vllm_rbln.patches.patch_registry import register_patch

register_patch(
    target="vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod",
    reason=(
        "RBLN needs a custom MXFP4 MoE method because upstream uses "
        "Mxfp4MoEMethod for non-XPU platforms but does not expose an "
        "out-of-tree kernel interface."
    ),
    owner_module=__name__,
)(PatchedMxfp4MoEMethod)
