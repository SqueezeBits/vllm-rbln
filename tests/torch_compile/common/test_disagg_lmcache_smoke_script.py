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

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _load_smoke_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = (
        repo_root
        / "examples"
        / "experimental"
        / "disaggregation"
        / "run_disagg_lmcache_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("run_disagg_lmcache_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_args(**overrides):
    base = {
        "role": "both",
        "model": "meta-llama/Llama-3.2-1B",
        "connector": "LMCacheConnectorV1",
        "lmcache_config_file": "",
        "shared_storage_path": "/tmp/rbln_lmcache_host_shared",
        "strict_lmcache": False,
        "require_lmcache_hit": False,
        "min_lmcache_hit_tokens": 1,
        "engine_id": "",
        "producer_rpc_port": "producer1",
        "consumer_rpc_port": "consumer1",
        "discard_partial_chunks": False,
        "prompt": "x",
        "prompt_repeat": 1,
        "prefill_max_tokens": 1,
        "decode_max_tokens": 1,
        "max_model_len": 128,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "block_size": 16,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": False,
        "enforce_eager": False,
        "phase_timeout_sec": 10,
        "result_json": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_transfer_config_uses_explicit_engine_id():
    module = _load_smoke_module()
    args = _make_args(engine_id="shared-engine")

    kv_transfer_config = module._build_transfer_config(args, "prefill")

    assert kv_transfer_config.engine_id == "shared-engine"


def test_build_subprocess_cmd_includes_engine_id_flag():
    module = _load_smoke_module()
    args = _make_args(engine_id="shared-engine")

    cmd = module._build_subprocess_cmd(args, "decode", Path("/tmp/result.json"))

    assert "--engine-id" in cmd
    idx = cmd.index("--engine-id")
    assert cmd[idx + 1] == "shared-engine"


def test_run_both_generates_one_engine_id_for_prefill_and_decode(monkeypatch):
    module = _load_smoke_module()
    args = _make_args(engine_id="")
    captured_engine_ids: list[str] = []

    def _fake_validate(_):
        return None

    def _fake_build_subprocess_cmd(local_args, role, result_json):
        captured_engine_ids.append(local_args.engine_id)
        return [role, str(result_json)]

    def _fake_subprocess_run(cmd, check, timeout, env, capture_output=None, text=None):
        role = cmd[0]
        result_json = cmd[1]
        payload = {
            "phase": role,
            "elapsed_sec": 0.01,
            "output_token_count": 1,
            "output_text_chars": 1,
            "finish_reason": "length",
        }
        Path(result_json).write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "_validate_lmcache_env", _fake_validate)
    monkeypatch.setattr(module, "_build_subprocess_cmd", _fake_build_subprocess_cmd)
    monkeypatch.setattr(module.subprocess, "run", _fake_subprocess_run)

    assert module._run_both(args) == 0
    assert len(captured_engine_ids) == 2
    assert captured_engine_ids[0] != ""
    assert captured_engine_ids[0] == captured_engine_ids[1]


def test_extract_lmcache_hit_tokens_uses_max_value():
    module = _load_smoke_module()
    logs = (
        "LMCache hit tokens: 0\n"
        "something\n"
        "LMCache hit tokens: 12\n"
        "LMCache hit tokens: 4\n"
    )
    assert module._extract_lmcache_hit_tokens(logs) == 12


def test_run_both_raises_when_require_lmcache_hit_and_no_hits(monkeypatch):
    module = _load_smoke_module()
    args = _make_args(require_lmcache_hit=True, min_lmcache_hit_tokens=1)

    def _fake_validate(_):
        return None

    def _fake_subprocess_run(cmd, check, timeout, env, capture_output=None, text=None):
        role = cmd[cmd.index("--role") + 1]
        result_json = cmd[cmd.index("--result-json") + 1]
        payload = {
            "phase": role,
            "elapsed_sec": 0.01,
            "output_token_count": 1,
            "output_text_chars": 1,
            "finish_reason": "length",
        }
        Path(result_json).write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout="LMCache hit tokens: 0\n",
            stderr="",
        )

    monkeypatch.setattr(module, "_validate_lmcache_env", _fake_validate)
    monkeypatch.setattr(module.subprocess, "run", _fake_subprocess_run)

    try:
        module._run_both(args)
    except RuntimeError as exc:
        assert "LMCache hit tokens" in str(exc)
    else:
        raise AssertionError("Expected strict LMCache hit requirement to fail.")


def test_run_both_accepts_when_require_lmcache_hit_and_hits_present(monkeypatch):
    module = _load_smoke_module()
    args = _make_args(require_lmcache_hit=True, min_lmcache_hit_tokens=1)

    def _fake_validate(_):
        return None

    def _fake_subprocess_run(cmd, check, timeout, env, capture_output=None, text=None):
        role = cmd[cmd.index("--role") + 1]
        result_json = cmd[cmd.index("--result-json") + 1]
        payload = {
            "phase": role,
            "elapsed_sec": 0.01,
            "output_token_count": 1,
            "output_text_chars": 1,
            "finish_reason": "length",
        }
        Path(result_json).write_text(json.dumps(payload), encoding="utf-8")
        hit_tokens = 3 if role == "decode" else 0
        return SimpleNamespace(
            returncode=0,
            stdout=f"LMCache hit tokens: {hit_tokens}\n",
            stderr="",
        )

    monkeypatch.setattr(module, "_validate_lmcache_env", _fake_validate)
    monkeypatch.setattr(module.subprocess, "run", _fake_subprocess_run)

    assert module._run_both(args) == 0
