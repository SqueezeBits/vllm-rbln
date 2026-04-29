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

import pytest


@pytest.fixture(autouse=True)
def reset_patch_registry_state(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())
    monkeypatch.setattr(patch_registry, "_general_extensions_loaded", False)


def _build_descriptor(
    *,
    key: str,
    owner_module: str,
    target: str | None = None,
    priority: int = 50,
):
    from vllm_rbln.patches.patch_registry import PatchDescriptor

    return PatchDescriptor(
        key=key,
        owner_module=owner_module,
        target=target or "vllm_rbln.patches.patch_registry._TEST_PATCH_SLOT",
        replacement=object(),
        reason="test descriptor",
        priority=priority,
    )


def test_patch_registry_separates_general_extensions_from_patch_descriptors():
    from vllm_rbln.patches.patch_registry import (
        get_general_extension_modules,
    )

    general_extensions = set(get_general_extension_modules())

    assert "vllm_rbln.distributed.kv_transfer.kv_connector.factory" in (
        general_extensions
    )
    assert "vllm_rbln.model_executor.layers.fused_moe.custom_ops" in (
        general_extensions
    )


def test_apply_patch_descriptors_is_idempotent(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry
    from vllm_rbln.patches.patch_registry import (
        PatchDescriptor,
        apply_patch_descriptors,
    )

    applied: list[str] = []

    descriptor = PatchDescriptor(
        key="test.patch_registry.idempotent",
        owner_module="tests.patch_registry",
        target="vllm_rbln.patches.patch_registry._TEST_PATCH_SLOT",
        replacement=object(),
        reason="test descriptor",
    )

    monkeypatch.setattr(
        patch_registry,
        "_apply_target_patch",
        lambda _: applied.append("called"),
    )
    monkeypatch.setattr(
        patch_registry,
        "_verify_target_patch",
        lambda _: None,
    )

    apply_patch_descriptors((descriptor,))
    apply_patch_descriptors((descriptor,))

    assert applied == ["called"]


def test_patch_descriptors_sort_by_priority_then_key():
    from vllm_rbln.patches.patch_registry import _sort_patch_descriptors

    descriptors = (
        _build_descriptor(key="b.default", owner_module="owner.b"),
        _build_descriptor(key="a.default", owner_module="owner.a"),
        _build_descriptor(key="z.high", owner_module="owner.z", priority=49),
    )

    assert [descriptor.key for descriptor in _sort_patch_descriptors(descriptors)] == [
        "z.high",
        "a.default",
        "b.default",
    ]


@pytest.mark.parametrize(
    ("general_extensions", "descriptors", "message"),
    [
        (
            ("general.module",),
            (
                _build_descriptor(
                    key="duplicate.key",
                    owner_module="owner.module.one",
                ),
                _build_descriptor(
                    key="duplicate.key",
                    owner_module="owner.module.two",
                    target="vllm_rbln.patches.patch_registry._TEST_PATCH_SLOT_TWO",
                ),
            ),
            "duplicate patch descriptor key: duplicate.key",
        ),
        (
            ("general.module",),
            (
                _build_descriptor(
                    key="patch.key.one",
                    owner_module="owner.module.one",
                ),
                _build_descriptor(
                    key="patch.key.two",
                    owner_module="owner.module.two",
                ),
            ),
            "duplicate patch descriptor target: vllm_rbln.patches.patch_registry._TEST_PATCH_SLOT",  # noqa: E501
        ),
        (
            ("owner.module",),
            (
                _build_descriptor(
                    key="patch.key",
                    owner_module="owner.module",
                ),
            ),
            "patch owner module must not be listed as a general extension: owner.module",  # noqa: E501
        ),
        (
            (),
            (
                _build_descriptor(
                    key="priority.low",
                    owner_module="owner.module",
                    priority=-1,
                ),
            ),
            "patch descriptor priority must be between 0 and 100: priority.low=-1",
        ),
        (
            (),
            (
                _build_descriptor(
                    key="priority.high",
                    owner_module="owner.module",
                    priority=101,
                ),
            ),
            "patch descriptor priority must be between 0 and 100: priority.high=101",
        ),
    ],
)
def test_validate_registry_layout_rejects_invalid_configs(
    monkeypatch,
    general_extensions,
    descriptors,
    message,
):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(
        patch_registry, "_GENERAL_EXTENSION_MODULES", general_extensions
    )
    monkeypatch.setattr(
        patch_registry, "_REGISTERED_PATCH_DESCRIPTORS", list(descriptors)
    )

    with pytest.raises(ValueError, match=message):
        patch_registry._validate_registry_layout()


def test_apply_patch_descriptors_runs_verify_once(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry
    from vllm_rbln.patches.patch_registry import (
        PatchDescriptor,
        apply_patch_descriptors,
    )

    events: list[str] = []
    descriptor = PatchDescriptor(
        key="test.patch_registry.verify",
        owner_module="tests.patch_registry.verify",
        target="vllm_rbln.patches.patch_registry._TEST_PATCH_SLOT",
        replacement=object(),
        reason="test descriptor",
        verify=lambda: events.append("verify"),
    )

    monkeypatch.setattr(
        patch_registry,
        "_apply_target_patch",
        lambda _: events.append("apply"),
    )

    apply_patch_descriptors((descriptor,))
    apply_patch_descriptors((descriptor,))

    assert events == ["apply", "verify"]


def test_apply_patch_descriptors_skips_when_condition_is_false(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry
    from vllm_rbln.patches.patch_registry import (
        PatchDescriptor,
        apply_patch_descriptors,
    )

    events: list[str] = []
    debug_calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        patch_registry.logger,
        "debug",
        lambda message, *args: debug_calls.append((message, args)),
    )

    descriptor = PatchDescriptor(
        key="test.patch_registry.condition",
        owner_module="tests.patch_registry.condition",
        target="vllm_rbln.patches.patch_registry._TEST_PATCH_SLOT",
        replacement=object(),
        reason="test descriptor",
        condition=lambda: False,
        verify=lambda: events.append("verify"),
    )

    monkeypatch.setattr(
        patch_registry,
        "_apply_target_patch",
        lambda _: events.append("apply"),
    )

    apply_patch_descriptors((descriptor,))

    assert events == []
    assert debug_calls == []


def test_register_general_extensions_imports_only_once(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    imported: list[tuple[str, ...]] = []
    modules = ("general.one", "general.two")

    monkeypatch.setattr(patch_registry, "_GENERAL_EXTENSION_MODULES", modules)
    monkeypatch.setattr(
        patch_registry,
        "_import_modules",
        lambda module_names, *, kind: imported.append(tuple(module_names)),
    )

    patch_registry.register_general_extensions()
    patch_registry.register_general_extensions()

    assert imported == [modules]


def test_apply_patch_descriptors_logs_when_patch_is_enabled(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry
    from vllm_rbln.patches.patch_registry import (
        PatchDescriptor,
        apply_patch_descriptors,
    )

    debug_calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        patch_registry.logger,
        "debug",
        lambda message, *args: debug_calls.append((message, args)),
    )
    monkeypatch.setattr(patch_registry, "_verify_target_patch", lambda _: None)

    descriptor = PatchDescriptor(
        key="test.patch_registry.logging",
        owner_module="tests.patch_registry.logging",
        target="vllm_rbln.patches.patch_registry._TEST_PATCH_SLOT",
        replacement=object(),
        reason="test descriptor",
    )

    apply_patch_descriptors((descriptor,))

    assert debug_calls == [
        (
            "Applying RBLN patch %s (owner=%s, target=%s)",
            (
                "test.patch_registry.logging",
                "tests.patch_registry.logging",
                "vllm_rbln.patches.patch_registry._TEST_PATCH_SLOT",
            ),
        )
    ]


def test_register_general_extensions_logs_when_modules_are_loaded(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    modules = ("general.one", "general.two")
    debug_calls: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(patch_registry, "_GENERAL_EXTENSION_MODULES", modules)
    monkeypatch.setattr(
        patch_registry,
        "_general_extensions_loaded",
        False,
    )
    monkeypatch.setattr(patch_registry, "import_module", lambda module_name: None)
    monkeypatch.setattr(
        patch_registry.logger,
        "debug",
        lambda message, *args: debug_calls.append((message, args)),
    )

    patch_registry.register_general_extensions()

    assert debug_calls == [
        ("Enabling %s via module import: %s", ("general extension", "general.one")),
        ("Enabling %s via module import: %s", ("general extension", "general.two")),
    ]
