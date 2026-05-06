# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Upstream contract tests for ``WorkerBase`` and ``RBLNWorker``.

These tests are intended to fail loudly when upstream ``WorkerBase`` changes in
ways that require a corresponding RBLN worker update.
"""

import inspect
from collections.abc import Callable

import pytest
from vllm.v1.worker.worker_base import WorkerBase

from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

EXPECTED_INHERITED_DEFAULT_METHODS = {
    "apply_model",
    "get_model_inspection",
    "reset_mm_cache",
}
EXPECTED_INHERITED_NOT_IMPLEMENTED_METHODS = {
    "get_cache_block_size_bytes",
}


def _get_public_worker_base_methods() -> dict[str, Callable[..., object]]:
    methods: dict[str, Callable[..., object]] = {}
    for name, obj in inspect.getmembers(WorkerBase, predicate=inspect.isfunction):
        if name.startswith("_") and name != "__init__":
            continue
        methods[name] = obj
    return methods


def _get_inherited_worker_base_methods() -> set[str]:
    return {
        name
        for name, base_method in _get_public_worker_base_methods().items()
        if getattr(RBLNWorker, name) is base_method
    }


def _get_worker_base_not_implemented_methods() -> set[str]:
    methods = set()
    for name, obj in _get_public_worker_base_methods().items():
        if "NotImplementedError" in inspect.getsource(obj):
            methods.add(name)
    return methods


def _erase_annotations(signature: inspect.Signature) -> inspect.Signature:
    params = [
        parameter.replace(annotation=inspect.Signature.empty)
        for parameter in signature.parameters.values()
    ]
    return signature.replace(
        parameters=params,
        return_annotation=inspect.Signature.empty,
    )


def test_rbln_worker_matches_worker_base_method_signatures() -> None:
    mismatches = []

    for name, base_method in _get_public_worker_base_methods().items():
        worker_method = getattr(RBLNWorker, name, None)
        if worker_method is None:
            mismatches.append(f"{name}: missing on RBLNWorker")
            continue

        # WorkerBase intentionally degrades some annotations to `object` at runtime
        # outside TYPE_CHECKING. Compare the callable shape here and rely on the
        # allowlist tests below to catch upstream contract drift.
        base_sig = _erase_annotations(inspect.signature(base_method))
        worker_sig = _erase_annotations(inspect.signature(worker_method))
        if worker_sig != base_sig:
            mismatches.append(f"{name}: base={base_sig}, worker={worker_sig}")

    assert mismatches == [], (
        "WorkerBase/RBLNWorker signature mismatch detected. "
        "If this changed due to an upstream update, update RBLNWorker to match:\n"
        + "\n".join(mismatches)
    )


def test_inherited_worker_base_methods_are_allowlisted() -> None:
    inherited_methods = _get_inherited_worker_base_methods()
    expected = (
        EXPECTED_INHERITED_DEFAULT_METHODS | EXPECTED_INHERITED_NOT_IMPLEMENTED_METHODS
    )

    assert inherited_methods == expected, (
        "The set of WorkerBase methods inherited by RBLNWorker changed. "
        "If this changed due to an upstream update, either implement the new "
        "method in RBLNWorker or explicitly classify it as an intentional "
        f"default-inherited method.\nexpected={sorted(expected)}\n"
        f"actual={sorted(inherited_methods)}"
    )


def test_inherited_not_implemented_worker_base_methods_are_allowlisted() -> None:
    inherited_not_implemented = (
        _get_inherited_worker_base_methods()
        & _get_worker_base_not_implemented_methods()
    )

    assert inherited_not_implemented == EXPECTED_INHERITED_NOT_IMPLEMENTED_METHODS, (
        "The set of inherited WorkerBase methods whose base implementation "
        "still raises NotImplementedError changed. This is an upstream contract "
        "change signal and needs triage.\n"
        f"expected={sorted(EXPECTED_INHERITED_NOT_IMPLEMENTED_METHODS)}\n"
        f"actual={sorted(inherited_not_implemented)}"
    )


def test_inherited_not_implemented_methods_raise_not_implemented_error() -> None:
    worker = object.__new__(RBLNWorker)

    for name in sorted(EXPECTED_INHERITED_NOT_IMPLEMENTED_METHODS):
        method = getattr(worker, name)
        with pytest.raises(NotImplementedError):
            method()
