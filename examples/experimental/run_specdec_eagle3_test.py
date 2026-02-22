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
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("RBLN_USE_CUSTOM_KERNEL", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_RBLN_USE_VLLM_MODEL", "1")
os.environ.setdefault("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("VLLM_RBLN_ENABLE_WARM_UP", "1")
# vLLM(v0.10.2) bug: speculative decoding works only in multi-processing.
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

# MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_ID = "Qwen/Qwen3-1.7B"
EAGLE_MODEL_ID = "AngelSlim/Qwen3-1.7B_eagle3"
RANDOM_EAGLE_ADAPTER_DIR = "/tmp/rbln_random_eagle3_llama32_1b"
NUM_SPECULATIVE_TOKENS = 3
MAX_MODEL_LEN = 1024
MAX_NUM_BATCHED_TOKENS = 64
MAX_NUM_SEQS = 2
DEFAULT_PROMPTS = ["The capital of France is", "A robot may not injure a human being"]
DEFAULT_MAX_TOKENS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Eagle3 speculative decoding on vLLM-RBLN. "
            "By default, a random local adapter is generated."
        )
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--eagle-model-id", default=EAGLE_MODEL_ID)
    parser.add_argument(
        "--use-random-adapter",
        action="store_true",
        help="Use a generated local random Eagle3 adapter.",
    )
    parser.add_argument("--random-adapter-dir", default=RANDOM_EAGLE_ADAPTER_DIR)
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=NUM_SPECULATIVE_TOKENS,
    )
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--block-size", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=MAX_NUM_BATCHED_TOKENS,
    )
    parser.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="One or more prompts to generate from.",
    )
    return parser.parse_args()


def ensure_random_eagle3_adapter(model_id: str, adapter_dir: str) -> str:
    adapter_path = Path(adapter_dir)
    config_path = adapter_path / "config.json"
    weights_path = adapter_path / "model.safetensors"
    if config_path.exists() and weights_path.exists():
        return str(adapter_path)

    adapter_path.mkdir(parents=True, exist_ok=True)

    # Keep base model config and add Eagle3-related fields.
    base_config = AutoConfig.from_pretrained(model_id)
    config_dict = base_config.to_dict()
    config_dict["draft_vocab_size"] = int(base_config.vocab_size)
    config_dict["target_hidden_size"] = int(base_config.hidden_size)
    config_dict["norm_before_residual"] = True
    config_dict["eagle_config"] = {"use_aux_hidden_state": True}

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Identity token-id mapping for draft vocabulary.
    draft_to_target = torch.arange(int(base_config.vocab_size), dtype=torch.long)
    save_file({"d2t": draft_to_target}, str(weights_path))
    return str(adapter_path)


def main() -> None:
    args = parse_args()

    use_random_adapter = args.use_random_adapter or not args.eagle_model_id
    if use_random_adapter:
        eagle_model_id = ensure_random_eagle3_adapter(
            args.model_id, args.random_adapter_dir
        )
    else:
        eagle_model_id = args.eagle_model_id

    adapter_mode = "random(local)" if use_random_adapter else "configured"
    print(f"EAGLE3 adapter mode: {adapter_mode}")
    print(f"EAGLE3 adapter path: {eagle_model_id}")

    # Create an LLM.
    llm = LLM(
        model=args.model_id,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        speculative_config={
            "method": "eagle3",
            "model": eagle_model_id,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(args.prompts, sampling_params=sampling_params)

    for output in outputs:
        print("-" * 50)
        print(f"prompt: {output.prompt}")
        print(f"generated text: {output.outputs[0].text}")
        print("-" * 50)

    try:
        metrics = llm.get_metrics()
    except AssertionError:
        print("Failed to load metrics.")
        return

    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * args.num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            num_positions = min(len(metric.values), len(acceptance_counts))
            for pos in range(num_positions):
                acceptance_counts[pos] += metric.values[pos]

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # print acceptance at each token position
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {i}: {acceptance_rate:.2f}")


if __name__ == "__main__":
    main()
