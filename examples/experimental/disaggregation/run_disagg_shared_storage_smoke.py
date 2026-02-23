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
"""Smoke test for disaggregated prefill/decode with SharedStorageConnector.

Usage:
    python3 examples/experimental/disaggregation/run_disagg_shared_storage_smoke.py \
        --model meta-llama/Llama-3.2-1B
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

os.environ.setdefault("RBLN_USE_CUSTOM_KERNEL", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_RBLN_USE_VLLM_MODEL", "1")
os.environ.setdefault("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("VLLM_RBLN_ENABLE_WARM_UP", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RBLN disaggregation smoke test via SharedStorageConnector"
    )
    parser.add_argument(
        "--role",
        choices=["both", "prefill", "decode"],
        default="both",
        help="Run both phases or a single phase.",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--shared-storage-path",
        type=str,
        default="/tmp/rbln_disagg_shared_storage",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain how KV cache reduces autoregressive decoding latency.",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=32,
        help="Repeat count to produce a non-trivial prefill prompt length.",
    )
    parser.add_argument("--prefill-max-tokens", type=int, default=1)
    parser.add_argument("--decode-max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--phase-timeout-sec",
        type=int,
        default=1800,
        help="Timeout for each subprocess phase when --role=both.",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default="",
        help="Optional output path used in single-phase mode.",
    )
    return parser.parse_args()


def _build_prompt(args: argparse.Namespace) -> str:
    return " ".join([args.prompt] * args.prompt_repeat)


def _build_transfer_config(args: argparse.Namespace, role: str) -> KVTransferConfig:
    kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
    connector_extra_config = {"shared_storage_path": args.shared_storage_path}
    return KVTransferConfig(
        kv_connector="SharedStorageConnector",
        kv_role=kv_role,
        kv_connector_extra_config=connector_extra_config,
    )


def _run_phase(args: argparse.Namespace, role: str) -> dict[str, Any]:
    Path(args.shared_storage_path).mkdir(parents=True, exist_ok=True)

    prompt = _build_prompt(args)
    max_tokens = args.prefill_max_tokens if role == "prefill" else args.decode_max_tokens

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        block_size=args.block_size,
        enable_chunked_prefill=True,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_transfer_config=_build_transfer_config(args, role),
    )

    start = time.perf_counter()
    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=max_tokens))
    elapsed_sec = time.perf_counter() - start

    if not outputs or not outputs[0].outputs:
        raise RuntimeError(f"{role} phase produced no outputs.")

    generated = outputs[0].outputs[0]
    token_count = len(generated.token_ids)
    if token_count < 1:
        raise RuntimeError(
            f"{role} phase generated zero tokens. Disaggregation path is not healthy."
        )

    result = {
        "phase": role,
        "elapsed_sec": elapsed_sec,
        "output_token_count": token_count,
        "output_text_chars": len(generated.text),
        "finish_reason": generated.finish_reason,
    }

    del llm
    gc.collect()

    return result


def _write_json(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_subprocess_cmd(
    args: argparse.Namespace, role: str, result_json: Path
) -> list[str]:
    cmd = [
        sys.executable,
        __file__,
        "--role",
        role,
        "--model",
        args.model,
        "--shared-storage-path",
        args.shared_storage_path,
        "--prompt",
        args.prompt,
        "--prompt-repeat",
        str(args.prompt_repeat),
        "--prefill-max-tokens",
        str(args.prefill_max_tokens),
        "--decode-max-tokens",
        str(args.decode_max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--block-size",
        str(args.block_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--result-json",
        str(result_json),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    return cmd


def _run_both(args: argparse.Namespace) -> int:
    Path(args.shared_storage_path).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rbln_disagg_shared_storage_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        prefill_json = tmp_path / "prefill.json"
        decode_json = tmp_path / "decode.json"

        prefill_cmd = _build_subprocess_cmd(args, "prefill", prefill_json)
        decode_cmd = _build_subprocess_cmd(args, "decode", decode_json)

        subprocess.run(prefill_cmd, check=True, timeout=args.phase_timeout_sec)
        subprocess.run(decode_cmd, check=True, timeout=args.phase_timeout_sec)

        prefill_result = json.loads(prefill_json.read_text(encoding="utf-8"))
        decode_result = json.loads(decode_json.read_text(encoding="utf-8"))

        if prefill_result["output_token_count"] < 1:
            raise RuntimeError("Prefill phase did not generate tokens.")
        if decode_result["output_token_count"] < 1:
            raise RuntimeError("Decode phase did not generate tokens.")

        summary = {
            "connector": "SharedStorageConnector",
            "prefill": prefill_result,
            "decode": decode_result,
        }
        print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    args = parse_args()

    if args.role == "both":
        return _run_both(args)

    result = _run_phase(args, args.role)
    _write_json(args.result_json, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
