# Disaggregation Smoke Scripts

This folder contains executable smoke scripts for prefill-decode disaggregation bring-up.

## Execution Context
- Follow the same smoke-test process used in `AGENTS.md`.
- Keep editing code in your local checkout, then run smoke commands through
  Docker container `tsk-24-27`.
- Use repository path in container: `/workspace/codes/vllm-rbln`.

## Scripts
- `run_disagg_shared_storage_smoke.py`
  - Baseline disaggregation using `SharedStorageConnector`.
- `run_disagg_lmcache_smoke.py`
  - LMCache-target disaggregation using `LMCacheConnectorV1`.

## TDD Workflow (Required)
1. Add or update a failing automated test first (`tests/v1/core`, `tests/v1/worker`).
2. Implement the feature change.
3. Re-run tests and confirm completion.
4. Run one of the smoke scripts here to verify end-to-end runtime behavior.

Do not use fixed/hardcoded outputs to make tests pass.

## Shared Storage Smoke
```bash
docker exec tsk-24-27 bash -lc '
  cd /workspace/codes/vllm-rbln && \
  python3 examples/experimental/disaggregation/run_disagg_shared_storage_smoke.py \
    --model meta-llama/Llama-3.2-1B \
    --shared-storage-path /tmp/rbln_disagg_shared_storage
'
```

Expected success signals:
- The script exits with code `0`.
- JSON summary includes both `prefill` and `decode`.
- `output_token_count >= 1` for both phases.

## LMCache Smoke
```bash
docker exec tsk-24-27 bash -lc '
  cd /workspace/codes/vllm-rbln && \
  python3 examples/experimental/disaggregation/run_disagg_lmcache_smoke.py \
    --model meta-llama/Llama-3.2-1B \
    --shared-storage-path /tmp/rbln_lmcache_host_shared
'
```

Notes:
- On RBLN, `LMCacheConnectorV1` is currently routed to host fallback connector `RBLNHostConnector`.
- This smoke script defaults `VLLM_RBLN_ENABLE_WARM_UP=0` and `VLLM_RBLN_COMPILE_MODEL=0` for stability.

Strict LMCache mode (no fallback):
```bash
docker exec tsk-24-27 bash -lc '
  cd /workspace/codes/vllm-rbln && \
  python3 examples/experimental/disaggregation/run_disagg_lmcache_smoke.py \
    --model meta-llama/Llama-3.2-1B \
    --strict-lmcache \
    --require-lmcache-hit \
    --min-lmcache-hit-tokens 1 \
    --decode-max-tokens 8
'
```

Prefill batch with different input lengths (repeat-count stride):
```bash
docker exec tsk-24-27 bash -lc '
  cd /workspace/codes/vllm-rbln && \
  python3 examples/experimental/disaggregation/run_disagg_lmcache_smoke.py \
    --role prefill \
    --model meta-llama/Llama-3.2-1B \
    --strict-lmcache \
    --prefill-batch-size 3 \
    --prefill-repeat-stride 2 \
    --max-num-seqs 3
'
```

Decode-all-requests strict smoke:
```bash
docker exec tsk-24-27 bash -lc '
  cd /workspace/codes/vllm-rbln && \
  python3 examples/experimental/disaggregation/run_disagg_lmcache_smoke.py \
    --role both \
    --model meta-llama/Llama-3.2-1B \
    --strict-lmcache \
    --require-lmcache-hit \
    --min-lmcache-hit-tokens 1 \
    --decode-max-tokens 8 \
    --prefill-batch-size 2
'
```

In strict mode, RBLN uses `RBLNLMCacheConnectorV1` (module:
`vllm_rbln.v1.kv_connector.rbln_lmcache_connector`) to avoid adapter
assertions when LMCache engine initialization degrades on non-CUDA devices.
For `--role both`, the script now auto-generates and reuses one shared
`engine_id` across prefill/decode subprocesses. You can override it with
`--engine-id <id>`.
By default, strict mode uses
`examples/experimental/disaggregation/lmcache_shared_backend.yaml` when
`LMCACHE_CONFIG_FILE` is not already set. This sample config enables
shared local-disk backend plus deterministic hash
(`pre_caching_hash_algorithm: "sha256_cbor"`).
You can still override config explicitly with:
`--lmcache-config-file <path>`.

Important backend note:
- If your LMCache config uses only `local_cpu` storage, cache data is
  process-local and decode can show `LMCache hit tokens: 0` even with shared
  `engine_id`.
- For true cross-process prefill->decode cache reuse, configure an LMCache
  backend that is shared across processes (for example remote/shared storage
  according to your LMCache deployment).
- Decode now always processes all provided requests. On current RBLN decode
  backend constraints, multi-request decode is executed sequentially per
  request in the script (instead of one kernel batch) to preserve correctness.

Expected success signals:
- The script exits with code `0`.
- JSON summary includes both `prefill` and `decode`.
- `output_token_count >= 1` for both phases.
- When running strict command above, `lmcache_hit_tokens.decode >= 1`
  (validated value is typically `256` in this smoke setup).

## Single-Phase Mode
Both scripts support:
- `--role prefill`
- `--role decode`

This is useful when producer and consumer are launched separately.
