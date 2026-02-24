# vLLM-RBLN: Torch Compile-Based Integration

## Introduction

`vllm-rbln` is a hardware plugin for the vLLM ecosystem that enables
high-throughput, low-latency LLM serving on RBLN NPUs.

This page is the `torch.compile`-based `vllm-rbln` overview. It describes the native vLLM execution path with RBLN NPUs where:

- You pass a Hugging Face model ID directly to `LLM(...)`.
- vLLM-RBLN compiles graphs during startup/warm-up.
- Compiled artifacts are cached and reused on subsequent runs.

Compared with the legacy `optimum-rbln` path, this architecture removes the separate offline compile/export step and aligns better with standard vLLM usage patterns.

### Quick Enablement

```bash
export VLLM_RBLN_USE_VLLM_MODEL=1
```

## Design Overview

The torch-compile path uses vLLM-native model loading and execution, with RBLN-specific runtime hooks and kernels registered through the plugin.

Operationally:

1. First startup is a cold start: model graphs are compiled.
2. Warm-up runs prefill/decode/sampler signatures automatically during engine initialization.
3. Later startups can become warm starts by reusing compile cache artifacts.

By default, compile artifacts are stored under:

- `$VLLM_CACHE_ROOT/rbln` (typically `~/.cache/vllm/rbln`)

You can force fresh compilation for debugging:

```bash
export VLLM_DISABLE_COMPILE_CACHE=1
```

## V1 Engine Support

The torch-compile path is built for vLLM V1 runtime execution.

- V1 is the default for current vLLM-RBLN torch-compile workflows.
- Pooling-model paths are V1-oriented.

## Model Serving Tutorials

The torch-compile path is organized into model-oriented serving tracks and feature-oriented serving tracks.

### Model Tutorials

- [Serving meta-llama/Llama-3.2-1B-Instruct with vLLM-RBLN (torch.compile)](./torch_compile_tutorial.md)

### Common Tutorial Pattern

Each torch-compile tutorial follows the same lifecycle:

1. Set torch-compile path env vars (`VLLM_RBLN_USE_VLLM_MODEL=1`).
2. Initialize `LLM(...)` (or start `vllm serve`) so startup automatically runs compile/warm-up.
3. Run normal offline inference or `vllm serve` OpenAI-compatible serving.
4. Reuse compile cache for warm start behavior.

## Supported Features

### 1) Single-Adapter LoRA

#### What is supported

- LoRA is available in the torch-compile runtime (`enable_lora=True`).
- Runtime behavior in the prefill path currently enforces single-adapter usage per prefill request flow.

#### Recommended serving profile

- `max_loras=1`
- `max_cpu_loras=1`
- Set an explicit `max_lora_rank` that matches your adapter family.

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
    max_loras=1,
    max_cpu_loras=1,
    max_lora_rank=8,
)
```

#### Practical note

Support for multiple LoRA adapters under torch-compile path is under development. Keep this path constrained to one active adapter per prefill trajectory to avoid behavior mismatches with current prefill-time assumptions.

### 2) Diverse Parallelism (TP / PP / EP)

The torch-compile path directly supports vLLM parallelism controls.

#### Tensor Parallelism (TP)

- Control via `tensor_parallel_size`.
- Used for splitting model compute across multiple devices/ranks.

#### Pipeline Parallelism (PP)

- Control via `pipeline_parallel_size`.
- Used to partition layers into stage-wise execution.

#### Expert Parallelism (EP)

- Enable via `enable_expert_parallel=True` for MoE models.
- Combine with TP/PP based on model architecture and NPU topology.

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen1.5-MoE-A2.7B",
    tensor_parallel_size=4,
    pipeline_parallel_size=1,
    enable_expert_parallel=True,
)
```

#### Operational note

When model parallelism is enabled (TP/PP/DP/EP), the platform config automatically adjusts distributed prerequisites internally. Keep serving configuration stable between warm-up and production traffic for best compile cache reuse.

### 3) Dynamic Batching with Bucketing

Dynamic decode batching is paired with bucketing to avoid graph churn.

#### Behavior

- Enabled automatically when `VLLM_RBLN_USE_VLLM_MODEL=1`.
- Incoming decode batch size is rounded up to a configured bucket.
- Runtime pads to the selected bucket and reuses compiled decode graphs.

This improves graph reuse and reduces recompiles caused by fluctuating request counts.

#### Main knobs

- `VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY`
  - `exponential` / `exp`
  - `linear`
  - `manual`
- `VLLM_RBLN_DECODE_BATCH_BUCKET_MIN`
- `VLLM_RBLN_DECODE_BATCH_BUCKET_STEP`
- `VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT`
- `VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS` (manual mode)

Example:

```bash
export VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY=exp
export VLLM_RBLN_DECODE_BATCH_BUCKET_MIN=2
export VLLM_RBLN_DECODE_BATCH_BUCKET_STEP=2
export VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT=8
```

For deeper behavior and bucket generation semantics, see `./bucketing.md`.

### 4) Quantization Support: W4A16 Group / W8A16 Channel

The torch-compile runtime includes mixed-precision linear kernel handling for common WnA16 serving patterns.

#### Supported quantization forms

- W4A16 group-wise quantization
- W8A16 channel-wise quantization

#### Kernel-level constraints

From the current mixed-precision kernel path:

- Supported weight types:
  - `uint4b8` (W4 family)
  - `uint8b128` (W8 family)
- Supported group-size modes:
  - `64`
  - `128`
  - `-1` (per-channel style mode)
- Symmetric quantization only:
  - `zero_points=False`
- Group/dynamic activation ordering (`g_idx`) is not supported.

#### Practical interpretation

- Use group-size-backed configurations for W4A16 group quant flows.
- Use per-channel style configuration (`group_size=-1`) for W8A16 channel-oriented flows.
- Ensure your checkpoint/quantization metadata matches the runtime kernel constraints above.

## Environment Variables 

| Variable | Default / Source | Behavior in this codebase | Key paths |
| --- | --- | --- | --- |
| `RBLN_USE_CUSTOM_KERNEL` | `False` (mapped to `VLLM_RBLN_USE_CUSTOM_KERNEL`) | Toggles kernel/op selection paths and some tensor allocation behavior. In attention backend, it switches between `rbln_triton_ops` and `rbln_custom_ops` calls. In KV-cache allocation, it switches cache tensor device between `cpu` and `meta`. | `vllm_rbln/rbln_envs.py`, `vllm_rbln/v1/attention/backends/flash_attention.py`, `vllm_rbln/v1/worker/rbln_model_runner.py` |
| `VLLM_RBLN_USE_VLLM_MODEL` | `False` | Master switch between torch-compile path (`1`) and legacy optimum path (`0`). Affects model/ops registration, worker/scheduler class selection, and runtime feature path. | `vllm_rbln/rbln_envs.py`, `vllm_rbln/__init__.py`, `vllm_rbln/platform.py` |
| `VLLM_RBLN_COMPILE_STRICT_MODE` | `False` | Adds `mode=\"strict\"` to `torch.compile` options for model and sampler compile paths. Useful for strict graph behavior and testing. | `vllm_rbln/rbln_envs.py`, `vllm_rbln/v1/worker/rbln_model_runner.py`, `vllm_rbln/v1/sample/rbln_sampler.py` |
| `VLLM_DISABLE_COMPILE_CACHE` | From upstream vLLM envs | Disables compile-cache reuse by preventing `cache_dir` from being set in compile options. If enabled, each run recompiles instead of reusing cached binaries. | `vllm_rbln/v1/worker/rbln_model_runner.py` |
| `VLLM_RBLN_ENABLE_WARM_UP` | `True` | Controls startup warm-up execution. If `0`, worker skips warm-up. For speculative decoding, platform currently forces warm-up off with warning. | `vllm_rbln/rbln_envs.py`, `vllm_rbln/v1/worker/rbln_worker.py`, `vllm_rbln/v1/worker/optimum_worker.py`, `vllm_rbln/platform.py` |
| `VLLM_RBLN_MOE_USE_OPT_KERNEL` | `True` | Selects the optimized MoE forward/custom-op branch. If `0`, code falls back to `VLLM_RBLN_MOE_CUSTOM_KERNEL` branch, then to native PyTorch MoE path. This choice is done during module import, so set it before process startup. | `vllm_rbln/rbln_envs.py`, `vllm_rbln/model_executor/layers/fused_moe/layer.py` |
| `RBLN_PROFILER` | `False` (mapped to `VLLM_RBLN_PROFILER`) | Exposed as an RBLN profiler flag. In this repo, it is primarily validated for incompatibility with multiprocessing, where startup raises an error. `VLLM_ENABLE_V1_MULTIPROCESSING=0` is required for the profiler to function. | `vllm_rbln/rbln_envs.py`, `vllm_rbln/platform.py`, `rebel/core/compiled_model_core.py` |

## Serving Checklist (Torch-Compile Path)

Before benchmarking or production rollout, verify the following:

1. `VLLM_RBLN_USE_VLLM_MODEL=1` is set in the serving environment.
2. Engine startup warm-up uses the same shape-sensitive serving knobs you will use in production (`max_num_seqs`, `max_num_batched_tokens`, `block_size`, etc.).
3. Decode bucket strategy is explicitly set (or intentionally left at defaults).
4. If LoRA is enabled, adapter concurrency aligns with single-adapter prefill behavior.
5. Quantized checkpoints satisfy kernel constraints (W4/W8 type, group settings, symmetric quantization).

Following this checklist minimizes unexpected recompilation and improves cold-to-
warm startup consistency.
