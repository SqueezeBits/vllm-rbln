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

from collections.abc import Iterable

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer, logger

from vllm_rbln.patches.patch_registry import register_patch

# NOTE(RBLN): This patch originated from
# https://github.com/RBLN-SW/vllm-rbln/pull/81 while enabling TP logits/model
# loading paths for the RBLN vLLM-model path.


@register_patch(
    target="vllm.model_executor.models.utils.AutoWeightsLoader._load_module",
    reason=(
        "The RBLN path needs tied embed_tokens weights to be replayed through "
        "the LM-head loading path when tensor parallelism is enabled, "
        "because token embeddings stay replicated while ParallelLMHead "
        "remains tensor-parallel sharded."
    ),
)
def auto_weights_loader_load_module(
    self: AutoWeightsLoader,
    base_prefix: str,
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[str]:
    if isinstance(module, PPMissingLayer):
        return

    # Avoid infinite recursion since this function is typically
    # called inside load_weights of the module itself
    if module != self.module:
        module_load_weights = getattr(module, "load_weights", None)
        if callable(module_load_weights):
            loaded_params = module_load_weights(weights)
            if loaded_params is None:
                logger.warning(
                    "Unable to collect loaded parameters for module %s", module
                )
            else:
                yield from map(
                    lambda x: self._get_qualname(base_prefix, x),
                    loaded_params,
                )

    child_modules = dict(module.named_children())
    child_params = dict(module.named_parameters(recurse=False))

    # Add missing tensors the weight loader needs to be able to load
    # that aren't registered as params, e.g., batchnorm statistics.
    self._add_loadable_non_param_tensors(module, child_params)

    embed_tokens_name = "embed_tokens"
    lm_head_name = "lm_head"
    tie_word_embeddings = any(p.startswith(lm_head_name) for p in self.skip_prefixes)
    tp_enabled = get_tensor_model_parallel_world_size() > 1

    embed_tokens: list[tuple[str, torch.Tensor]] = []

    def gen_weights(cur_weights: Iterable[tuple[str, torch.Tensor]]):
        for name, weight in cur_weights:
            if name.startswith(embed_tokens_name):
                new_name = name.replace(embed_tokens_name, lm_head_name)
                embed_tokens.append((new_name, weight))
            yield (name, weight)

    for child_prefix, child_weights in self._groupby_prefix(weights):
        prefix = self._get_qualname(base_prefix, child_prefix)

        if child_prefix in child_modules:
            if self._can_skip(prefix + "."):
                logger.debug("Skipping module %s", prefix)

                continue

            if tie_word_embeddings and tp_enabled:
                child_weights = gen_weights(child_weights)
            yield from self._load_module(
                prefix, child_modules[child_prefix], child_weights
            )
        elif child_prefix in child_params:
            if self._can_skip(prefix):
                logger.debug("Skipping param %s", prefix)

                continue

            yield from self._load_param(
                prefix, child_params[child_prefix], child_weights
            )
        else:
            can_skip_module = self._can_skip(prefix + ".")
            can_skip_param = self._can_skip(prefix)
            if can_skip_module or can_skip_param:
                logger.debug("Skipping missing %s", prefix)

                continue

            can_ignore_module = self._can_ignore_unexpected(prefix + ".")
            can_ignore_param = self._can_ignore_unexpected(prefix)
            if can_ignore_module or can_ignore_param:
                logger.debug("Ignoring missing %s", prefix)

                continue

            msg = (
                f"There is no module or parameter named '{prefix}' "
                f"in {type(self.module).__name__}"
            )
            raise ValueError(msg)

    if embed_tokens:
        org_skip_prefixes = self.skip_prefixes
        self.skip_prefixes = [
            p for p in org_skip_prefixes if not p.startswith(lm_head_name)
        ]

        for child_prefix, child_weights in self._groupby_prefix(embed_tokens):
            assert child_prefix == lm_head_name
            prefix = self._get_qualname(base_prefix, child_prefix)
            if child_prefix in child_modules:
                yield from self._load_module(
                    prefix,
                    child_modules[child_prefix],
                    child_weights,
                )

        self.skip_prefixes = org_skip_prefixes
