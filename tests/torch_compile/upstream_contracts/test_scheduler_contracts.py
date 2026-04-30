# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Upstream contract tests for scheduler boundaries used by RBLNScheduler.

These tests intentionally check the interfaces RBLNScheduler depends on, not the
source body of upstream Scheduler.schedule().
"""

import inspect
from collections.abc import Callable

import pytest
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_rbln.v1.core.rbln_kv_cache_manager import RBLNKVCacheManager
from vllm_rbln.v1.core.rbln_scheduler import RBLNScheduler, RBLNSchedulerOutput


def _parameter_names(callable_obj: Callable[..., object]) -> list[str]:
    return list(inspect.signature(callable_obj).parameters)


def _assert_parameters_include(
    callable_obj: Callable[..., object],
    expected_names: set[str],
) -> None:
    actual_names = set(_parameter_names(callable_obj))

    missing = expected_names - actual_names
    assert missing == set(), (
        f"{callable_obj} is missing parameters required by RBLNScheduler: "
        f"{sorted(missing)}. actual={sorted(actual_names)}"
    )


# EngineCore calls schedule() without extra arguments, so RBLNScheduler must keep
# the same public entrypoint shape as upstream Scheduler.
def test_scheduler_schedule_signature_is_self_only() -> None:
    expected = ["self"]

    assert _parameter_names(Scheduler.schedule) == expected
    assert _parameter_names(RBLNScheduler.schedule) == expected


# EngineCore feeds SchedulerOutput and ModelRunnerOutput back through this hook;
# RBLNScheduler wraps upstream behavior and must keep the same callable shape.
def test_scheduler_update_from_output_parameter_names_match() -> None:
    expected = ["self", "scheduler_output", "model_runner_output"]

    assert _parameter_names(Scheduler.update_from_output) == expected
    assert _parameter_names(RBLNScheduler.update_from_output) == expected


# RBLN adds copy ops for sub-block caching, but the output must still satisfy
# the upstream SchedulerOutput contract consumed by vLLM.
def test_rbln_scheduler_output_extends_scheduler_output() -> None:
    assert issubclass(RBLNSchedulerOutput, SchedulerOutput)


# RBLNScheduler may use either upstream KVCacheManager or RBLNKVCacheManager
# depending on sub-block caching eligibility, so both must accept its call args.
@pytest.mark.parametrize("manager_cls", [KVCacheManager, RBLNKVCacheManager])
def test_kv_cache_managers_allocate_slots_support_rbln_scheduler_arguments(
    manager_cls: type[KVCacheManager],
) -> None:
    _assert_parameters_include(
        manager_cls.allocate_slots,
        {
            "self",
            "request",
            "num_new_tokens",
            "num_new_computed_tokens",
            "new_computed_blocks",
            "num_lookahead_tokens",
            "num_external_computed_tokens",
            "delay_cache_blocks",
            "num_encoder_tokens",
        },
    )


# RBLNScheduler delays cache mutation until schedule finalization and then calls
# cache_blocks() on whichever KV cache manager is active.
@pytest.mark.parametrize("manager_cls", [KVCacheManager, RBLNKVCacheManager])
def test_kv_cache_managers_cache_blocks_support_delayed_finalization(
    manager_cls: type[KVCacheManager],
) -> None:
    assert _parameter_names(manager_cls.cache_blocks) == [
        "self",
        "request",
        "num_computed_tokens",
    ]
