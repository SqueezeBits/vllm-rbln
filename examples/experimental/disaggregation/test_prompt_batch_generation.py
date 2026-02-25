# Copyright 2025 Rebellions Inc. All rights reserved.

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def _load_smoke_module():
    module_path = Path(__file__).with_name("run_disagg_lmcache_smoke.py")
    spec = importlib.util.spec_from_file_location("disagg_smoke", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_prompts_prefill_batch_with_repeat_stride():
    smoke = _load_smoke_module()
    args = SimpleNamespace(
        prompt="hello",
        prompt_repeat=2,
        prefill_batch_size=3,
        prefill_repeat_stride=2,
    )

    prompts = smoke._build_prompts(args)
    assert prompts == [
        "hello hello",
        "hello hello hello hello",
        "hello hello hello hello hello hello",
    ]


def test_build_prompts_decode_uses_full_batch():
    smoke = _load_smoke_module()
    args = SimpleNamespace(
        prompt="x",
        prompt_repeat=3,
        prefill_batch_size=3,
        prefill_repeat_stride=2,
    )

    prompts = smoke._build_prompts(args)
    assert prompts == [
        "x x x",
        "x x x x x",
        "x x x x x x x",
    ]


def test_generate_with_decode_fallback_processes_all_requests():
    smoke = _load_smoke_module()
    llm = Mock()
    sampling_params = object()

    llm.generate.side_effect = [
        ["decode-0"],
        ["decode-1"],
        ["decode-2"],
    ]
    outputs = smoke._generate_with_decode_fallback(
        llm=llm,
        prompts=["p0", "p1", "p2"],
        sampling_params=sampling_params,
        role="decode",
    )

    assert outputs == ["decode-0", "decode-1", "decode-2"]
    assert llm.generate.call_count == 3
    assert llm.generate.call_args_list[0].args == (["p0"], sampling_params)
    assert llm.generate.call_args_list[1].args == (["p1"], sampling_params)
    assert llm.generate.call_args_list[2].args == (["p2"], sampling_params)


def test_generate_with_decode_fallback_keeps_batched_prefill():
    smoke = _load_smoke_module()
    llm = Mock()
    sampling_params = object()

    llm.generate.return_value = ["prefill-batch"]
    outputs = smoke._generate_with_decode_fallback(
        llm=llm,
        prompts=["p0", "p1", "p2"],
        sampling_params=sampling_params,
        role="prefill",
    )

    assert outputs == ["prefill-batch"]
    llm.generate.assert_called_once_with(["p0", "p1", "p2"], sampling_params)


def test_effective_max_num_seqs_prefill_scales_with_prompt_count():
    smoke = _load_smoke_module()
    args = SimpleNamespace(max_num_seqs=1)

    effective = smoke._effective_max_num_seqs(
        args=args,
        role="prefill",
        num_prompts=3,
    )

    assert effective == 3


def test_effective_max_num_seqs_decode_uses_single_seq_with_fallback():
    smoke = _load_smoke_module()
    args = SimpleNamespace(max_num_seqs=1)

    effective = smoke._effective_max_num_seqs(
        args=args,
        role="decode",
        num_prompts=3,
    )

    assert effective == 1
