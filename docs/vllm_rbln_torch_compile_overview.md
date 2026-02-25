# vLLM-RBLN: Torch Compile Overview

## Introduction

`vllm-rbln` integrates RBLN NPUs into vLLM through the plugin system.

This page covers the **torch.compile path** only:

- Hugging Face model IDs are loaded directly in `LLM(...)`.
- Graph compilation and warm-up run during engine startup.
- Compiled artifacts are reused from cache on later runs.

Compared with the legacy `optimum-rbln` flow, this path removes the separate
offline export/compile step.

## Quick Start

```bash
export VLLM_RBLN_USE_VLLM_MODEL=1
```

## Runtime Behavior

1. Cold start: compile + warm-up happen during engine initialization.
2. Warm start: compile cache is reused.
3. Recompile triggers: shape-sensitive config changes (for example
   `max_num_seqs`, `max_num_batched_tokens`, `block_size`).

Force fresh compile:

```bash
export VLLM_DISABLE_COMPILE_CACHE=1
```

Disable warm-up:

```bash
export VLLM_RBLN_ENABLE_WARM_UP=0
```

## Model Serving Tutorials

- [Serving meta-llama/Llama-3.2-1B-Instruct with vLLM-RBLN (torch.compile)](./torch_compile_tutorial.md)

## Feature Guides

Detailed feature documentation is centralized in:

- [Torch Compile Feature Guides](./torch_compile_features.md)

Direct links:

- [Structured Output](./torch_compile_features.md#structured-output)
- [Automatic Prefix Caching](./torch_compile_features.md#automatic-prefix-caching)
- [Dynamic Batching (with Decode Bucketing)](./torch_compile_features.md#dynamic-batching-with-decode-bucketing)
- [OpenAI-Compatible API Serving](./torch_compile_features.md#openai-compatible-api-serving)
- [Custom Kernel Path](./torch_compile_features.md#custom-kernel-path)

## Environment Variables

Feature-specific flags are documented in:

- [Torch Compile Feature Guides](./torch_compile_features.md)
- [Decode Bucketing](./bucketing.md)

For full code-level environment variable definitions, see:

- `vllm_rbln/rbln_envs.py`
