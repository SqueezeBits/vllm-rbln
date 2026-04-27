# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class PatchDescriptor:
    key: str
    owner_module: str
    targets: tuple[str, ...]
    reason: str
    apply: Callable[[], None]
    assumptions: tuple[str, ...] = ()
    contract_tests: tuple[str, ...] = ()
    verify: Callable[[], None] | None = None
    order_hint: int = 0


# NOTE: These modules are part of bootstrap/registration, not upstream symbol
# replacement. They are kept separate from legacy patch modules so migration to
# registry-managed patches can proceed incrementally without obscuring intent.
_GENERAL_EXTENSION_MODULES: tuple[str, ...] = (
    "vllm_rbln.distributed.kv_transfer.kv_connector.factory",
    "vllm_rbln.triton_kernels.attention",
    "vllm_rbln.triton_kernels.causal_attention",
    "vllm_rbln.triton_kernels.flash_attention",
    "vllm_rbln.triton_kernels.flash_causal_attention",
    "vllm_rbln.triton_kernels.sliding_window_attention",
)


# NOTE: These modules still patch upstream symbols at import time. They remain
# on the legacy path until each cluster is migrated to explicit descriptors.
_LEGACY_PATCH_MODULES: tuple[str, ...] = (
    "vllm_rbln.model_executor.layers.attention.attention",
    "vllm_rbln.forward_context",
    "vllm_rbln.lora.layer",
    "vllm_rbln.model_executor.layers.fused_moe.layer",
    "vllm_rbln.model_executor.layers.fused_moe.shared_fused_moe",
    "vllm_rbln.model_executor.layers.logits_processor",
    "vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision",
    "vllm_rbln.model_executor.layers.quantization.mxfp4",
    "vllm_rbln.model_executor.layers.quantization.fp8",
    "vllm_rbln.model_executor.layers.rotary_embedding.base",
    "vllm_rbln.model_executor.layers.rotary_embedding.deepseek_scaling_rope",
    "vllm_rbln.model_executor.layers.vocab_parallel_embedding",
    "vllm_rbln.model_executor.model_loader.weight_loader",
    "vllm_rbln.models.deepseek_v2",
    "vllm_rbln.models.gpt_oss",
    "vllm_rbln.models.qwen2",
    "vllm_rbln.models.qwen2_moe",
    "vllm_rbln.models.qwen3",
    "vllm_rbln.models.qwen3_moe",
    "vllm_rbln.models.minimax_m2",
    "vllm_rbln.models.utils",
)


# NOTE: The bridge registry starts empty. Patch clusters will move here
# incrementally from `_LEGACY_PATCH_MODULES`.
_REGISTERED_PATCH_DESCRIPTORS: tuple[PatchDescriptor, ...] = ()

_applied_patch_keys: set[str] = set()
_general_extensions_loaded = False
_legacy_patch_modules_loaded = False


def _import_modules(module_names: Iterable[str], *, kind: str) -> None:
    for module_name in module_names:
        logger.debug("Enabling %s via module import: %s", kind, module_name)
        import_module(module_name)


def _sort_patch_descriptors(
    descriptors: Sequence[PatchDescriptor],
) -> tuple[PatchDescriptor, ...]:
    return tuple(sorted(descriptors, key=lambda d: (d.order_hint, d.key)))


def _validate_registry_layout() -> None:
    general_extensions = set(_GENERAL_EXTENSION_MODULES)
    legacy_patches = set(_LEGACY_PATCH_MODULES)
    if not general_extensions.isdisjoint(legacy_patches):
        msg = "general extension modules and legacy patch modules must be disjoint"
        raise ValueError(msg)

    descriptor_keys: set[str] = set()
    descriptor_owner_modules: set[str] = set()
    for descriptor in _REGISTERED_PATCH_DESCRIPTORS:
        if descriptor.key in descriptor_keys:
            msg = f"duplicate patch descriptor key: {descriptor.key}"
            raise ValueError(msg)
        descriptor_keys.add(descriptor.key)

        if descriptor.owner_module in descriptor_owner_modules:
            msg = f"duplicate patch descriptor owner module: {descriptor.owner_module}"
            raise ValueError(msg)
        descriptor_owner_modules.add(descriptor.owner_module)

        if descriptor.owner_module in general_extensions:
            msg = (
                "registry-managed patch owner module must not be listed as a "
                f"general extension: {descriptor.owner_module}"
            )
            raise ValueError(msg)

        if descriptor.owner_module in legacy_patches:
            msg = (
                "registry-managed patch owner module must not remain in the "
                f"legacy path: {descriptor.owner_module}"
            )
            raise ValueError(msg)


def get_general_extension_modules() -> tuple[str, ...]:
    return _GENERAL_EXTENSION_MODULES


def get_legacy_patch_modules() -> tuple[str, ...]:
    return _LEGACY_PATCH_MODULES


def get_registered_patch_descriptors() -> tuple[PatchDescriptor, ...]:
    return _sort_patch_descriptors(_REGISTERED_PATCH_DESCRIPTORS)


def register_general_extensions() -> None:
    global _general_extensions_loaded

    if _general_extensions_loaded:
        return

    _import_modules(_GENERAL_EXTENSION_MODULES, kind="general extension")
    _general_extensions_loaded = True


def apply_patch_descriptors(descriptors: Sequence[PatchDescriptor]) -> None:
    for descriptor in _sort_patch_descriptors(descriptors):
        if descriptor.key in _applied_patch_keys:
            continue

        logger.debug(
            "Enabling registry-managed patch %s (owner=%s, targets=%s)",
            descriptor.key,
            descriptor.owner_module,
            ", ".join(descriptor.targets),
        )
        descriptor.apply()
        if descriptor.verify is not None:
            descriptor.verify()
        _applied_patch_keys.add(descriptor.key)


def apply_registered_patches() -> None:
    apply_patch_descriptors(get_registered_patch_descriptors())


def import_legacy_patch_modules() -> None:
    global _legacy_patch_modules_loaded

    if _legacy_patch_modules_loaded:
        return

    _import_modules(_LEGACY_PATCH_MODULES, kind="legacy patch")
    _legacy_patch_modules_loaded = True


_validate_registry_layout()


__all__ = (
    "PatchDescriptor",
    "apply_patch_descriptors",
    "apply_registered_patches",
    "get_general_extension_modules",
    "get_legacy_patch_modules",
    "get_registered_patch_descriptors",
    "import_legacy_patch_modules",
    "register_general_extensions",
)
