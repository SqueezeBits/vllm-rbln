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
    monkeypatch.setattr(patch_registry, "_legacy_patch_modules_loaded", False)


def _build_descriptor(*, key: str, owner_module: str):
    from vllm_rbln.patches.patch_registry import PatchDescriptor

    return PatchDescriptor(
        key=key,
        owner_module=owner_module,
        targets=("vllm.fake.symbol",),
        reason="test descriptor",
        apply=lambda: None,
    )


def test_patch_registry_splits_general_extensions_from_legacy_patches():
    from vllm_rbln.patches.patch_registry import (
        get_general_extension_modules,
        get_legacy_patch_modules,
    )

    general_extensions = set(get_general_extension_modules())
    legacy_patches = set(get_legacy_patch_modules())

    assert "vllm_rbln.distributed.kv_transfer.kv_connector.factory" in (
        general_extensions
    )
    assert "vllm_rbln.model_executor.layers.attention.attention" in legacy_patches
    assert general_extensions.isdisjoint(legacy_patches)


def test_apply_patch_descriptors_is_idempotent():
    from vllm_rbln.patches.patch_registry import (
        PatchDescriptor,
        apply_patch_descriptors,
    )

    applied: list[str] = []

    descriptor = PatchDescriptor(
        key="test.patch_registry.idempotent",
        owner_module="tests.patch_registry",
        targets=("vllm.fake.symbol",),
        reason="test descriptor",
        apply=lambda: applied.append("called"),
    )

    apply_patch_descriptors((descriptor,))
    apply_patch_descriptors((descriptor,))

    assert applied == ["called"]


@pytest.mark.parametrize(
    ("general_extensions", "legacy_patches", "descriptors", "message"),
    [
        (
            ("shared.module",),
            ("shared.module",),
            (),
            "general extension modules and legacy patch modules must be disjoint",
        ),
        (
            ("general.module",),
            (),
            (
                _build_descriptor(
                    key="duplicate.key",
                    owner_module="owner.module.one",
                ),
                _build_descriptor(
                    key="duplicate.key",
                    owner_module="owner.module.two",
                ),
            ),
            "duplicate patch descriptor key: duplicate.key",
        ),
        (
            ("general.module",),
            (),
            (
                _build_descriptor(
                    key="patch.key.one",
                    owner_module="owner.module",
                ),
                _build_descriptor(
                    key="patch.key.two",
                    owner_module="owner.module",
                ),
            ),
            "duplicate patch descriptor owner module: owner.module",
        ),
        (
            ("owner.module",),
            (),
            (
                _build_descriptor(
                    key="patch.key",
                    owner_module="owner.module",
                ),
            ),
            "registry-managed patch owner module must not be listed as a general extension: owner.module",  # noqa: E501
        ),
        (
            (),
            ("owner.module",),
            (
                _build_descriptor(
                    key="patch.key",
                    owner_module="owner.module",
                ),
            ),
            "registry-managed patch owner module must not remain in the legacy path: owner.module",  # noqa: E501
        ),
    ],
)
def test_validate_registry_layout_rejects_invalid_configs(
    monkeypatch,
    general_extensions,
    legacy_patches,
    descriptors,
    message,
):
    import vllm_rbln.patches.patch_registry as patch_registry

    monkeypatch.setattr(
        patch_registry, "_GENERAL_EXTENSION_MODULES", general_extensions
    )
    monkeypatch.setattr(patch_registry, "_LEGACY_PATCH_MODULES", legacy_patches)
    monkeypatch.setattr(patch_registry, "_REGISTERED_PATCH_DESCRIPTORS", descriptors)

    with pytest.raises(ValueError, match=message):
        patch_registry._validate_registry_layout()


def test_apply_patch_descriptors_runs_verify_once(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry
    from vllm_rbln.patches.patch_registry import (
        PatchDescriptor,
        apply_patch_descriptors,
    )

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())

    events: list[str] = []
    descriptor = PatchDescriptor(
        key="test.patch_registry.verify",
        owner_module="tests.patch_registry.verify",
        targets=("vllm.fake.symbol",),
        reason="test descriptor",
        apply=lambda: events.append("apply"),
        verify=lambda: events.append("verify"),
    )

    apply_patch_descriptors((descriptor,))
    apply_patch_descriptors((descriptor,))

    assert events == ["apply", "verify"]


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


def test_import_legacy_patch_modules_imports_only_once(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    imported: list[tuple[str, ...]] = []
    modules = ("legacy.one", "legacy.two")

    monkeypatch.setattr(patch_registry, "_LEGACY_PATCH_MODULES", modules)
    monkeypatch.setattr(
        patch_registry,
        "_import_modules",
        lambda module_names, *, kind: imported.append(tuple(module_names)),
    )

    patch_registry.import_legacy_patch_modules()
    patch_registry.import_legacy_patch_modules()

    assert imported == [modules]


def test_apply_patch_descriptors_logs_when_patch_is_enabled(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry
    from vllm_rbln.patches.patch_registry import (
        PatchDescriptor,
        apply_patch_descriptors,
    )

    monkeypatch.setattr(patch_registry, "_applied_patch_keys", set())

    debug_calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        patch_registry.logger,
        "debug",
        lambda message, *args: debug_calls.append((message, args)),
    )

    descriptor = PatchDescriptor(
        key="test.patch_registry.logging",
        owner_module="tests.patch_registry.logging",
        targets=("vllm.fake.symbol",),
        reason="test descriptor",
        apply=lambda: None,
    )

    apply_patch_descriptors((descriptor,))

    assert debug_calls == [
        (
            "Enabling registry-managed patch %s (owner=%s, targets=%s)",
            (
                "test.patch_registry.logging",
                "tests.patch_registry.logging",
                "vllm.fake.symbol",
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


def test_import_legacy_patch_modules_logs_when_modules_are_loaded(monkeypatch):
    import vllm_rbln.patches.patch_registry as patch_registry

    modules = ("legacy.one", "legacy.two")
    debug_calls: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(patch_registry, "_LEGACY_PATCH_MODULES", modules)
    monkeypatch.setattr(
        patch_registry,
        "_legacy_patch_modules_loaded",
        False,
    )
    monkeypatch.setattr(patch_registry, "import_module", lambda module_name: None)
    monkeypatch.setattr(
        patch_registry.logger,
        "debug",
        lambda message, *args: debug_calls.append((message, args)),
    )

    patch_registry.import_legacy_patch_modules()

    assert debug_calls == [
        ("Enabling %s via module import: %s", ("legacy patch", "legacy.one")),
        ("Enabling %s via module import: %s", ("legacy patch", "legacy.two")),
    ]
