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

from typing import ClassVar

import torch


class LoRAInputs:
    sampler_indices_padded: ClassVar[torch.Tensor]

    @classmethod
    def set_sampler_indices_padded(cls, sampler_indices_padded: torch.Tensor) -> None:
        cls.sampler_indices_padded = sampler_indices_padded

    @classmethod
    def get_sampler_indices_padded(cls) -> torch.Tensor:
        return cls.sampler_indices_padded


def create_sampler_indices_padded(
    lora_ids: list[int],
    lora_index_to_id: list[int],
    max_num_seqs: int,
    is_prefill: bool,
    max_loras: int,
    device: torch.device,
) -> torch.Tensor:
    if is_prefill:
        assert len(lora_ids) == 1, "Only single LoRA is supported during prefill phase"

    prompt_mapping: list[int] = [
        lora_index_to_id.index(lora_ids[i])
        if i < len(lora_ids) and lora_ids[i] > 0
        else -1
        for i in range(len(lora_ids) if is_prefill else max_num_seqs)
    ]
    sampler_indices_padded = torch.tensor(
        prompt_mapping, dtype=torch.long, device=device
    )
    sampler_indices_padded = torch.where(
        sampler_indices_padded == -1, max_loras, sampler_indices_padded
    )
    sampler_indices_padded = torch.arange(
        0, len(sampler_indices_padded), dtype=torch.long, device=device
    ) + (sampler_indices_padded * len(sampler_indices_padded))

    return sampler_indices_padded
