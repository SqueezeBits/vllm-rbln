# Torch Compile Feature Guides

This page summarizes major `vllm-rbln` features for the **torch.compile path only**.

Prerequisite for all sections:

```bash
export VLLM_RBLN_USE_VLLM_MODEL=1
```

## Structured Output

### Overview

Torch-compile workers support vLLM structured output by applying grammar bitmasks
to logits before sampling.

In this path:

- Requests in `WAITING_FOR_FSM` state are deferred until grammar/FSM is ready.
- The grammar bitmask is applied in `sample_tokens()` before token sampling.

### Example (Offline API)

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    max_model_len=4096,
    max_num_seqs=8,
    block_size=1024,
    enable_chunked_prefill=True,
)

structured = StructuredOutputsParams(
    json={"sentiment": "str", "score": "float"}
)
params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
    structured_outputs=structured,
)

out = llm.generate(["Analyze: I love this movie."], params)
print(out[0].outputs[0].text)
```

### Notes

- First structured-output request may include FSM compilation latency.
- This behavior is part of the standard vLLM V1 structured-output flow, executed
  through the torch-compile worker/scheduler path.

## Automatic Prefix Caching

### Overview

Prefix caching works through the vLLM KV-cache manager/scheduler path and is
automatically leveraged when enabled.

For torch-compile path specifically:

- Prefix caching is auto-disabled for sliding-window models.
- Other model types follow normal vLLM prefix-caching behavior.

### Example (Offline API)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_prefix_caching=True,
    max_model_len=4096,
    max_num_seqs=8,
    block_size=1024,
    enable_chunked_prefill=True,
)

common_prefix = "You are a concise assistant. Answer in one sentence.\n\n"
prompts = [
    common_prefix + "Question: What is CUDA?",
    common_prefix + "Question: What is ROCm?",
]

out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
for r in out:
    print(r.outputs[0].text)
```

### Notes

- Reuse is strongest when prompts share long common prefixes.
- Keep serving shape/config stable to maximize compile and cache reuse together.

## Dynamic Batching (with Decode Bucketing)

### Overview

Torch-compile scheduling uses dynamic decode batching with decode-bucket
round-up/padding to reduce recompilation.

Behavior highlights:

- Prefill is constrained to batch size `1` in RBLN scheduler path.
- Decode batches are grouped into configured buckets.
- Bucketing is automatically enabled when `VLLM_RBLN_USE_VLLM_MODEL=1`.

### Configuration

```bash
export VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY=exp
export VLLM_RBLN_DECODE_BATCH_BUCKET_MIN=1
export VLLM_RBLN_DECODE_BATCH_BUCKET_STEP=2
export VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT=8
```

See `./bucketing.md` for strategy details.

### Notes

- Bucketed decode helps keep graph signatures reusable under variable traffic.
- If runtime shapes drift from warm-up signatures, additional compiles can occur.

## OpenAI-Compatible API Serving

### Overview

Torch-compile path supports vLLM OpenAI-compatible serving through
`vllm serve` with the RBLN plugin enabled.

### Example (Server + Request)

```bash
export VLLM_PLUGINS=rbln
export VLLM_RBLN_USE_VLLM_MODEL=1

vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --device rbln \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --block-size 1024 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 128
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role":"user","content":"Say hello in 5 words."}],
    "max_tokens": 32,
    "temperature": 0.0
  }'
```

### Notes

- First requests can include compile/warm-up latency.
- After warm start, cached artifacts reduce repeated compile overhead.

## Custom Kernel Path

### Overview

`RBLN_USE_CUSTOM_KERNEL` toggles kernel/op selection for torch-compile runtime
paths, including attention op dispatch and KV-cache allocation behavior.

### Enable

```bash
export RBLN_USE_CUSTOM_KERNEL=1
```

### Behavior

- Attention backend selects different op namespaces depending on this flag.
- KV-cache raw tensors switch allocation device (`cpu` vs `meta`) based on this
  flag in model-runner allocation path.

### Notes

- Set before process start so all startup-time initialization paths see it.
- Use with the torch-compile path (`VLLM_RBLN_USE_VLLM_MODEL=1`).
