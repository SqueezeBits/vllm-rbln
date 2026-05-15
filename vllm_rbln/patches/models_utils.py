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

from vllm_rbln.model_executor.models.utils import patched_load_module
from vllm_rbln.patches import register_patch

# NOTE(RBLN): Introduced in https://github.com/RBLN-SW/vllm-rbln/pull/81
register_patch(
    target="vllm.model_executor.models.utils.AutoWeightsLoader._load_module",
    reason=(
        "In RBLN tensor parallelism, tied word embeddings cannot alias weights "
        "because token embeddings are replicated while ParallelLMHead is "
        "vocab-sharded. Replay embed_tokens weights through the normal lm_head "
        "loading path so ParallelLMHead.weight_loader can load each rank-local "
        "vocab shard."
    ),
    owner_module=__name__,
)(patched_load_module)
