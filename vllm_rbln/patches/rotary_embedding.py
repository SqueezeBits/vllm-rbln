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

from vllm_rbln.model_executor.layers.rotary_embedding.base import (
    patched_rope_forward_oot,
    patched_rope_init,
)
from vllm_rbln.patches import register_patch

register_patch(
    target="vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.__init__",
    reason=(
        "The RBLN path needs rotate-half style RoPE caches because that "
        "cache layout is more compatible with RBLN execution than the "
        "upstream layout."
    ),
    owner_module=__name__,
)(patched_rope_init)


register_patch(
    target="vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.forward_oot",
    reason=(
        "The RBLN path needs a dedicated RoPE execution path that consumes "
        "the rotate-half cache layout prepared at initialization and "
        "applies rotary embeddings with RBLN-friendly tensor layouts."
    ),
    owner_module=__name__,
)(patched_rope_forward_oot)
