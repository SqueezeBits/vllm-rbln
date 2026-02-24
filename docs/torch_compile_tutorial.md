# Tutorial: vLLM-RBLN with torch.compile

This tutorial shows the `torch.compile`-based path for vLLM-RBLN.

Unlike the legacy `optimum-rbln` path, there is no separate offline compile step. You pass the Hugging Face model ID directly to vLLM, and vLLM-RBLN compiles and warms up the model automatically during engine startup.

## Setup

Create `inference.py`:

```python
from vllm import LLM, SamplingParams

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

prompts = [
    "The president of the United States is",
    "The capital of France is",
]

llm = LLM(
    model=MODEL_ID,
    max_model_len=4 * 1024,
    max_num_seqs=8,
    max_num_batched_tokens=128,
    block_size=1024,
)

outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

## Execution

Install dependencies:

```bash
pip install -U pip
pip install "vllm-rbln"
```

Set environment variables for the `torch.compile` path:

```bash
export VLLM_RBLN_USE_VLLM_MODEL=1
```

Run inference:

```bash
python inference.py
```

## Notes

- You do not need a separate warm-up script for standard usage.
- Compile/warm-up runs automatically at engine startup (for example when `LLM(...)` is initialized, or when `vllm serve` starts the engine).
- The first startup is slower because graphs are compiled and warm-up runs.
- Later runs reuse cached binaries by default (`$VLLM_CACHE_ROOT/rbln`, typically `~/.cache/vllm/rbln`).
- If you change model/runtime shape settings (for example `max_num_seqs`, `max_num_batched_tokens`, or `block_size`), additional compilation can occur.
- To disable warm-up explicitly:

```bash
export VLLM_RBLN_ENABLE_WARM_UP=0
```

- To force a fresh compile for debugging:

```bash
export VLLM_DISABLE_COMPILE_CACHE=1
```

For detailed variable behavior, see the `Environment Variables` section in `vllm_rbln_torch_compile_overview.md`.
