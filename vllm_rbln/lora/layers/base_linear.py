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
# https://github.com/RBLN-SW/vllm-rbln/pull/169 while adding initial LoRA
# support to the RBLN vLLM-model path.

import torch
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA


def patched_base_linear_apply(
    self: BaseLinearLayerWithLoRA,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
    output_org_shape = output.shape

    x = x.reshape(
        -1, x.shape[-1]
    )  # [bs, seq_len, hidden_size] -> [bs * seq_len, hidden_size]
    output = output.reshape(
        -1, output.shape[-1]
    )  # [bs, seq_len, hidden_size] -> [bs * seq_len, hidden_size]

    lora_output: torch.Tensor = self.punica_wrapper.add_lora_linear(
        output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices
    )

    return lora_output.view(output_org_shape)
