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
"""Smoke test for disaggregated prefill/decode with LMCacheConnectorV1.

Usage:
    python3 examples/experimental/disaggregation/run_disagg_lmcache_smoke.py \
        --model meta-llama/Llama-3.2-1B
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_RBLN_USE_VLLM_MODEL", "1")
# NOTE:
# LMCacheConnectorV1 is auto-routed to a host fallback connector on RBLN.
# Keep compile/warmup disabled by default for this smoke path to avoid known
# intermittent aborts in RBLN compile runtime during connector bring-up.
os.environ.setdefault("VLLM_RBLN_ENABLE_WARM_UP", "0")
os.environ.setdefault("VLLM_RBLN_COMPILE_MODEL", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RBLN disaggregation smoke test via LMCacheConnectorV1"
    )
    parser.add_argument(
        "--role",
        choices=["both", "prefill", "decode"],
        default="both",
        help="Run both phases or a single phase.",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--connector",
        type=str,
        default="LMCacheConnectorV1",
        help="KV connector name registered in vLLM.",
    )
    parser.add_argument(
        "--engine-id",
        type=str,
        default="",
        help=(
            "Optional shared KV transfer engine id. If omitted with "
            "--role both, one id is generated and reused for both phases."
        ),
    )
    parser.add_argument(
        "--lmcache-config-file",
        type=str,
        default="",
        help="Optional LMCache config file. If set, exports LMCACHE_CONFIG_FILE.",
    )
    parser.add_argument(
        "--shared-storage-path",
        type=str,
        default="/tmp/rbln_lmcache_host_shared",
        help=(
            "Host shared path used by RBLN LMCache fallback connector "
            "(RBLNHostConnector)."
        ),
    )
    parser.add_argument(
        "--strict-lmcache",
        action="store_true",
        help="Disable RBLN host fallback and force strict LMCacheConnectorV1 path.",
    )
    parser.add_argument(
        "--require-lmcache-hit",
        action="store_true",
        help=(
            "Fail smoke when decode phase does not report LMCache hit tokens. "
            "Use with strict mode and shared LMCache backend config."
        ),
    )
    parser.add_argument(
        "--min-lmcache-hit-tokens",
        type=int,
        default=1,
        help="Minimum LMCache hit token count required when --require-lmcache-hit is set.",
    )
    parser.add_argument(
        "--producer-rpc-port",
        type=str,
        default="producer1",
        help="LMCache RPC port key for producer role.",
    )
    parser.add_argument(
        "--consumer-rpc-port",
        type=str,
        default="consumer1",
        help="LMCache RPC port key for consumer role.",
    )
    parser.add_argument(
        "--discard-partial-chunks",
        action="store_true",
        help="Enable discard_partial_chunks in connector extra config.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Summarize why prefill-decode disaggregation improves throughput.",
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


def _validate_lmcache_env(args: argparse.Namespace) -> None:
    if "PYTHONHASHSEED" not in os.environ:
        # LMCache builtin hash must be stable across processes for cross-process
        # lookup/store behavior.
        os.environ["PYTHONHASHSEED"] = "0"

    if args.lmcache_config_file:
        os.environ["LMCACHE_CONFIG_FILE"] = args.lmcache_config_file

    if args.strict_lmcache:
        os.environ["VLLM_RBLN_STRICT_LMCACHE"] = "1"
        if "LMCACHE_CONFIG_FILE" not in os.environ:
            default_cfg = Path(__file__).with_name("lmcache_shared_backend.yaml")
            if default_cfg.exists():
                os.environ["LMCACHE_CONFIG_FILE"] = str(default_cfg)
            else:
                raise RuntimeError(
                    "Strict LMCache mode requires LMCACHE_CONFIG_FILE or "
                    "--lmcache-config-file."
                )

    if args.require_lmcache_hit and not args.strict_lmcache:
        raise RuntimeError("--require-lmcache-hit requires --strict-lmcache.")

    if args.require_lmcache_hit and args.min_lmcache_hit_tokens < 1:
        raise RuntimeError("--min-lmcache-hit-tokens must be >= 1.")


def _build_prompt(args: argparse.Namespace) -> str:
    return " ".join([args.prompt] * args.prompt_repeat)


def _build_transfer_config(args: argparse.Namespace, role: str) -> KVTransferConfig:
    kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
    rpc_port = args.producer_rpc_port if role == "prefill" else args.consumer_rpc_port
    connector_extra_config = {
        "discard_partial_chunks": args.discard_partial_chunks,
        "lmcache_rpc_port": rpc_port,
        "shared_storage_path": args.shared_storage_path,
    }
    kwargs: dict[str, Any] = {
        "kv_connector": args.connector,
        "kv_role": kv_role,
        "kv_connector_extra_config": connector_extra_config,
    }
    if args.engine_id:
        kwargs["engine_id"] = args.engine_id
    return KVTransferConfig(**kwargs)


def _run_phase(args: argparse.Namespace, role: str) -> dict[str, Any]:
    _validate_lmcache_env(args)

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


def _extract_lmcache_hit_tokens(log_text: str) -> int:
    matches = re.findall(r"LMCache hit tokens:\s*(\d+)", log_text or "")
    if not matches:
        return 0
    return max(int(m) for m in matches)


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
        "--connector",
        args.connector,
        "--engine-id",
        args.engine_id,
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
        "--producer-rpc-port",
        args.producer_rpc_port,
        "--consumer-rpc-port",
        args.consumer_rpc_port,
        "--shared-storage-path",
        args.shared_storage_path,
        "--result-json",
        str(result_json),
    ]
    if args.lmcache_config_file:
        cmd.extend(["--lmcache-config-file", args.lmcache_config_file])
    if args.strict_lmcache:
        cmd.append("--strict-lmcache")
    if args.discard_partial_chunks:
        cmd.append("--discard-partial-chunks")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    return cmd


def _run_both(args: argparse.Namespace) -> int:
    both_args = args
    if not args.engine_id:
        both_args = argparse.Namespace(**vars(args))
        both_args.engine_id = f"rbln-disagg-{uuid.uuid4()}"

    _validate_lmcache_env(both_args)
    with tempfile.TemporaryDirectory(prefix="rbln_disagg_lmcache_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        prefill_json = tmp_path / "prefill.json"
        decode_json = tmp_path / "decode.json"

        prefill_cmd = _build_subprocess_cmd(both_args, "prefill", prefill_json)
        decode_cmd = _build_subprocess_cmd(both_args, "decode", decode_json)

        env = os.environ.copy()
        if both_args.lmcache_config_file:
            env["LMCACHE_CONFIG_FILE"] = both_args.lmcache_config_file

        prefill_proc = subprocess.run(
            prefill_cmd,
            check=True,
            timeout=both_args.phase_timeout_sec,
            env=env,
            capture_output=True,
            text=True,
        )
        decode_proc = subprocess.run(
            decode_cmd,
            check=True,
            timeout=both_args.phase_timeout_sec,
            env=env,
            capture_output=True,
            text=True,
        )

        for proc in (prefill_proc, decode_proc):
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)

        prefill_hits = _extract_lmcache_hit_tokens(
            f"{prefill_proc.stdout}\n{prefill_proc.stderr}"
        )
        decode_hits = _extract_lmcache_hit_tokens(
            f"{decode_proc.stdout}\n{decode_proc.stderr}"
        )

        prefill_result = json.loads(prefill_json.read_text(encoding="utf-8"))
        decode_result = json.loads(decode_json.read_text(encoding="utf-8"))

        if prefill_result["output_token_count"] < 1:
            raise RuntimeError("Prefill phase did not generate tokens.")
        if decode_result["output_token_count"] < 1:
            raise RuntimeError("Decode phase did not generate tokens.")

        summary = {
            "connector": both_args.connector,
            "engine_id": both_args.engine_id,
            "lmcache_config_file": os.environ.get("LMCACHE_CONFIG_FILE", ""),
            "lmcache_hit_tokens": {
                "prefill": prefill_hits,
                "decode": decode_hits,
            },
            "prefill": prefill_result,
            "decode": decode_result,
        }
        if both_args.require_lmcache_hit and decode_hits < both_args.min_lmcache_hit_tokens:
            raise RuntimeError(
                "Strict LMCache smoke expected decode LMCache hit tokens >= "
                f"{both_args.min_lmcache_hit_tokens}, observed {decode_hits}. "
                "Ensure shared backend config and stable hashing "
                "(e.g. PYTHONHASHSEED=0)."
            )
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
