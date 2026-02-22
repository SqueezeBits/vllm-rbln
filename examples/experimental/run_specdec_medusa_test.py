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
os.environ.setdefault("VLLM_RBLN_COMPILE_STRICT_MODE", "0")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("VLLM_RBLN_ENABLE_WARM_UP", "0")
# vLLM(v0.10.2) bug: speculative decoding works only in multi-processing.
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

MODEL_ID = "meta-llama/Llama-3.2-1B"
MEDUSA_MODEL_ID = os.getenv("MEDUSA_MODEL_ID", "")
RANDOM_MEDUSA_ADAPTER_DIR = "/tmp/rbln_random_medusa_llama32_1b"
NUM_SPECULATIVE_TOKENS = 3
MAX_MODEL_LEN = 1024
MAX_NUM_BATCHED_TOKENS = 64
MAX_NUM_SEQS = 1
DEFAULT_PROMPTS = ["The capital of France is"]
DEFAULT_MAX_TOKENS = 4
DEFAULT_RANDOM_MEDUSA_LAYERS = 1
DEFAULT_TRUNCATED_VOCAB_SIZE = 2048
DEFAULT_RANDOM_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Medusa speculative decoding on vLLM-RBLN. "
            "By default, a random local Medusa adapter is generated."
        )
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--medusa-model-id", default=MEDUSA_MODEL_ID)
    parser.add_argument(
        "--use-random-adapter",
        action="store_true",
        help="Use a generated local random Medusa adapter.",
    )
    parser.add_argument("--random-adapter-dir", default=RANDOM_MEDUSA_ADAPTER_DIR)
    parser.add_argument(
        "--random-medusa-layers",
        type=int,
        default=DEFAULT_RANDOM_MEDUSA_LAYERS,
    )
    parser.add_argument(
        "--truncated-vocab-size",
        type=int,
        default=DEFAULT_TRUNCATED_VOCAB_SIZE,
    )
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


def _is_compatible_adapter(
    config_path: Path,
    *,
    hidden_size: int,
    vocab_size: int,
    num_speculative_tokens: int,
    num_medusa_layers: int,
    truncated_vocab_size: int,
) -> bool:
    try:
        config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        config_dict.get("model_type") == "medusa"
        and int(config_dict.get("hidden_size", -1)) == hidden_size
        and int(config_dict.get("vocab_size", -1)) == vocab_size
        and int(config_dict.get("num_heads", -1)) == num_speculative_tokens
        and int(config_dict.get("num_hidden_layers", -1)) == num_medusa_layers
        and int(config_dict.get("truncated_vocab_size", -1)) == truncated_vocab_size
        and bool(config_dict.get("original_lm_head", False))
    )


def ensure_random_medusa_adapter(
    model_id: str,
    adapter_dir: str,
    num_speculative_tokens: int,
    num_medusa_layers: int,
    truncated_vocab_size: int,
) -> str:
    base_config = AutoConfig.from_pretrained(model_id)
    hidden_size = int(base_config.hidden_size)
    vocab_size = int(base_config.vocab_size)

    if truncated_vocab_size <= 0 or truncated_vocab_size > vocab_size:
        raise ValueError(
            f"truncated_vocab_size must be in [1, {vocab_size}], "
            f"but got {truncated_vocab_size}"
        )
    if num_medusa_layers <= 0:
        raise ValueError(
            f"num_medusa_layers must be >= 1, but got {num_medusa_layers}"
        )

    adapter_path = Path(adapter_dir)
    config_path = adapter_path / "config.json"
    weights_path = adapter_path / "model.safetensors"

    if (
        config_path.exists()
        and weights_path.exists()
        and _is_compatible_adapter(
            config_path,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_speculative_tokens=num_speculative_tokens,
            num_medusa_layers=num_medusa_layers,
            truncated_vocab_size=truncated_vocab_size,
        )
    ):
        return str(adapter_path)

    adapter_path.mkdir(parents=True, exist_ok=True)

    medusa_config = {
        "model_type": "medusa",
        "architectures": ["MedusaModel"],
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_heads": num_speculative_tokens,
        "num_hidden_layers": num_medusa_layers,
        "truncated_vocab_size": truncated_vocab_size,
        "original_lm_head": True,
        "medusa_fc_bias": False,
    }
    config_path.write_text(json.dumps(medusa_config, indent=2), encoding="utf-8")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(DEFAULT_RANDOM_SEED)

    state_dict: dict[str, torch.Tensor] = {}
    for head_idx in range(num_speculative_tokens):
        for layer_idx in range(num_medusa_layers):
            weight_key = f"blocks.{head_idx}.layers.{layer_idx}.weight"
            state_dict[weight_key] = torch.randn(
                hidden_size,
                hidden_size,
                generator=generator,
                dtype=torch.bfloat16,
            ) * 1e-3

    state_dict["lm_head.weight"] = torch.randn(
        truncated_vocab_size,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    ) * 1e-3
    state_dict["token_map"] = torch.arange(truncated_vocab_size, dtype=torch.long)
    save_file(state_dict, str(weights_path))

    return str(adapter_path)


def main() -> None:
    args = parse_args()

    use_random_adapter = args.use_random_adapter or not args.medusa_model_id
    if use_random_adapter:
        medusa_model_id = ensure_random_medusa_adapter(
            model_id=args.model_id,
            adapter_dir=args.random_adapter_dir,
            num_speculative_tokens=args.num_speculative_tokens,
            num_medusa_layers=args.random_medusa_layers,
            truncated_vocab_size=args.truncated_vocab_size,
        )
    else:
        medusa_model_id = args.medusa_model_id

    adapter_mode = "random(local)" if use_random_adapter else "configured"
    print(f"MEDUSA adapter mode: {adapter_mode}")
    print(f"MEDUSA adapter path: {medusa_model_id}")

    # Create an LLM.
    llm = LLM(
        model=args.model_id,
        dtype=torch.bfloat16,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        speculative_config={
            "method": "medusa",
            "model": medusa_model_id,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
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
    acceptance_length = (
        1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    )
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # print acceptance at each token position
    for pos, accepted_at_pos in enumerate(acceptance_counts):
        acceptance_rate = accepted_at_pos / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {pos}: {acceptance_rate:.2f}")


if __name__ == "__main__":
    main()
