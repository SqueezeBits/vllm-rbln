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

from vllm_rbln.patches.patch_registry import (
    PatchDescriptor,
    apply_patch_descriptors,
    apply_registered_patches,
    get_general_extension_modules,
    get_legacy_patch_modules,
    get_registered_patch_descriptors,
    import_legacy_patch_modules,
    register_general_extensions,
    register_patch,
)

from . import (
    logits_processor,  # noqa: F401
    lora,  # noqa: F401
    vocab_parallel_embedding,  # noqa: F401
)

__all__ = (
    "PatchDescriptor",
    "apply_patch_descriptors",
    "apply_registered_patches",
    "get_general_extension_modules",
    "get_legacy_patch_modules",
    "get_registered_patch_descriptors",
    "import_legacy_patch_modules",
    "register_general_extensions",
    "register_patch",
)
