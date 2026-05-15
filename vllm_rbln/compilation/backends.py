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
from collections.abc import Sequence
from typing import Any

import torch
import torch.fx as fx
from rebel.core.torch_compile import rbln_backend as _rbln_backend

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


# TODO(RBLN): Implement RBLN-specific backend like VllmBackend
def rbln_backend(
    graph: fx.GraphModule, example_inputs: Sequence[Any], **kwargs: Any
) -> Any:
    parts = []
    for i in example_inputs:
        if isinstance(i, torch.Tensor):
            parts.append(f"{tuple(i.shape)}:{i.dtype}")
        else:
            parts.append(type(i).__name__)
    summarized_inputs = f"[{', '.join(parts)}]"
    logger.debug(
        "rbln_backend: inputs=%s",
        summarized_inputs,
    )
    return _rbln_backend(graph, example_inputs, **kwargs)
