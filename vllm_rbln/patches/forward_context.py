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

# NOTE(RBLN): This patch originated from
# https://github.com/RBLN-SW/vllm-rbln/commit/58577d6 while adding MoE PoC
# support, and was later updated through
# https://github.com/RBLN-SW/vllm-rbln/pull/192,
# https://github.com/RBLN-SW/vllm-rbln/pull/252,
# https://github.com/RBLN-SW/vllm-rbln/pull/293,
# https://github.com/RBLN-SW/vllm-rbln/commit/191b133, and
# https://github.com/RBLN-SW/vllm-rbln/pull/524.

from typing import Any

from vllm_rbln.forward_context import _set_forward_context
from vllm_rbln.patches.patch_registry import register_patch


@register_patch(
    target="vllm.forward_context.set_forward_context",
    reason=(
        "RBLN needs a custom forward context to carry RBLN DP padding "
        "metadata, MoE token-mask metadata, and KV-cache binding kwargs "
        "through each model forward pass."
    ),
)
def rbln_set_forward_context(*args: Any, **kwargs: Any):
    return _set_forward_context(*args, **kwargs)
