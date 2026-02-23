# Prefill-Decode Disaggregation Plan (RBLN + LMCache)

This document is the source of truth for implementing prefill-decode disaggregation support for RBLN NPUs with LMCache integration.

## Goals
- Enable disaggregated prefill/decode in the RBLN v1 path (`VLLM_RBLN_USE_VLLM_MODEL=1`).
- Support LMCache integration via vLLM KV connector (`LMCacheConnectorV1`).
- Preserve RBLN scheduling constraints (no mixed prefill/decode batching, prefill batch size 1).
- Validate behavior with automated tests and executable smoke scripts.

## Non-Goals
- No new support work for optimum scheduler path in this effort.
- No hardcoded output assertions or fake pass conditions.
- No bypassing correctness checks for benchmark-only gains.

## Mandatory Development Policy (TDD)
For every behavior change:
1. Add/modify a test that fails first.
2. Implement the minimal feature change to make that test pass.
3. Run the relevant suite and confirm pass.
4. Refactor only after tests are green.

Rules:
- Never write tests that pass because of fixed strings, mocked outputs that bypass logic, or disabled assertions.
- Tests must validate actual scheduler/worker/connector state transitions or runtime outputs.

## Current Baseline
- `vllm_rbln/v1/core/rbln_scheduler.py` already includes KV connector hooks (`get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta`) and `WAITING_FOR_REMOTE_KVS` handling.
- `vllm_rbln/v1/worker/rbln_model_runner.py` already integrates `KVConnectorModelRunnerMixin` and transfer-group registration.
- `vllm_rbln/v1/core/optimum_scheduler.py` does not currently implement full KV transfer flow.
- KV connector tests are missing in `tests/v1/core` and `tests/v1/worker`.
- `tests/torch_compile/v1/core/utils.py` explicitly disables KV connector tests today.

## Current Status (2026-02-23)
- Implemented:
  - RBLN KV connector alias normalization in `vllm_rbln/platform.py`.
  - Automatic routing from `LMCacheConnectorV1` -> `RBLNHostConnector` fallback on RBLN (unless `VLLM_RBLN_STRICT_LMCACHE=1`).
  - Strict connector `RBLNLMCacheConnectorV1` in `vllm_rbln/v1/kv_connector/rbln_lmcache_connector.py`:
    - routes strict mode through LMCache connector interfaces on RBLN.
    - avoids LMCache adapter assertions when LMCache engine is unavailable on non-CUDA devices.
    - degrades to safe no-op/recompute semantics instead of crashing.
    - fixes RBLN 6D paged KV read/write indexing in `RBLNHostGPUConnector`
      for host-backed LMCache transfer (`from_gpu`/`to_gpu`).
    - adds strict-wrapper forwarding for `register_kv_caches` so LMCache uses
      explicit KV cache registration instead of legacy forward-context fallback.
    - patches LMCache `LocalDiskBackend` on RBLN to lazily restore missing
      in-memory index entries from on-disk chunk files during `contains()`.
      This enables cross-process prefill->decode reuse when using shared
      local-disk backend and deterministic hash config.
  - `RBLNHostConnector` host-mediated load path (`RBLN -> host storage -> RBLN`) in `vllm_rbln/v1/kv_connector/rbln_host_connector.py`.
  - Worker KV handshake metadata path in `vllm_rbln/v1/worker/rbln_worker.py`.
  - No-forward KV execution guard in `vllm_rbln/v1/worker/rbln_model_runner.py`.
  - Step-2 worker transfer semantics coverage in
    `tests/torch_compile/common/test_rbln_worker_disaggregation.py` for:
    - no-forward decision with/without connector metadata.
    - PP connector output pass-through for empty/non-empty connector states.
    - propagation of `invalid_block_ids` and send/recv completion sets.
  - PP KV connector pass-through fix in `vllm_rbln/v1/worker/rbln_worker.py`:
    - switched emptiness check to `kv_connector_output.is_empty()` so
      non-empty connector payloads (for example `invalid_block_ids`) are
      not dropped when finished-send/recv sets are empty.
  - Example smoke scripts and README under `examples/experimental/disaggregation`.
  - LMCache smoke script hardening in
    `examples/experimental/disaggregation/run_disagg_lmcache_smoke.py`:
    - adds explicit `--engine-id` support.
    - auto-generates and reuses one shared engine id for prefill/decode when
      `--role both` is used.
  - Tests:
    - strict connector routing test in `tests/torch_compile/common/test_platform.py`.
    - strict connector degraded-mode guard tests in `tests/torch_compile/common/test_rbln_lmcache_connector.py`.
    - strict connector 6D KV-layout regression tests in
      `tests/torch_compile/common/test_rbln_lmcache_connector.py`.
    - strict-wrapper `register_kv_caches` delegation test in
      `tests/torch_compile/common/test_rbln_lmcache_connector.py`.
    - strict LocalDisk metadata/default-shape builder and disk-index restore
      tests in `tests/torch_compile/common/test_rbln_lmcache_connector.py`.
    - LMCache smoke script regression tests in
      `tests/torch_compile/common/test_disagg_lmcache_smoke_script.py` for:
      - explicit engine-id propagation to transfer config.
      - subprocess command propagation of engine-id.
      - shared engine-id generation across prefill/decode in `--role both`.
- Smoke validation:
  - Shared-storage disaggregation smoke passed.
  - LMCache-target smoke passed via host fallback path (compile/warmup disabled by default in script).
  - Strict LMCache smoke (`--strict-lmcache`) passed end-to-end with real generation for prefill and decode.
  - 6D KV-layout regression reproduced via failing connector test and fixed;
    post-fix strict LMCache smoke remains green.
  - strict-wrapper registration regression reproduced (failing delegation
    test) and fixed; strict LMCache runtime now logs `Registering KV caches`
    and no longer emits `Please update LMCacheConnector...` warning.
  - strict shared-backend LMCache smoke with
    `examples/experimental/disaggregation/lmcache_shared_backend.yaml`
    passed with non-zero decode hit tokens (`lmcache_hit_tokens.decode = 256`
    in validated runs).
  - Step-2 regression reproduced (failing test on dropped `invalid_block_ids`)
    and resolved (test passing after worker fix).
  - AGENTS-required SpecDec ngram smoke passed (`num_drafts > 0`, `num_accepted_tokens > 0`).
- Known constraints:
  - LMCache still expects CUDA/XPU-oriented adapter behavior upstream; RBLN
    strict mode relies on host-backed connector patching to run safely.
  - If strict LMCache engine initialization fails for a runtime/config
    combination, connector behavior degrades to recompute/no-op semantics.
  - Configs using only process-local `local_cpu` backend still show
    cross-process `LMCache hit tokens: 0`; use a shared backend config
    (for example shared `local_disk`) and deterministic hashing
    (`pre_caching_hash_algorithm: "sha256_cbor"`).
  - Compile-enabled LMCache smoke can still hit intermittent `librbln`
    malloc abort during warmup/compile.

## Implementation Phases

### Phase 0: Guardrails and Scope Clarity
Tasks:
- Add explicit warning/guard when KV transfer is requested on optimum path.
- Keep implementation target on `RBLNScheduler` and `RBLNModelRunner`.

Tests first:
- Add test that verifies optimum path with KV transfer raises/warns with clear message.

Done when:
- No silent fallback for unsupported optimum KV transfer.

### Phase 1: Scheduler Connector Parity
Tasks:
- Align `RBLNScheduler` connector lifecycle with upstream vLLM behavior for:
  - matched-token query
  - async recv waiting transitions
  - alloc/update connector state
  - metadata emission
- Validate interplay with RBLN prefill/decode scheduling constraints.

Tests first (core):
- New tests in `tests/v1/core` for:
  - `WAITING_FOR_REMOTE_KVS` request transitions to `WAITING`.
  - skipped requests re-queued correctly when remote KV not ready.
  - connector metadata is attached to scheduler output when connector is enabled.
  - no mixed prefill/decode batch regression with connector active.

Done when:
- All new scheduler connector tests pass.

### Phase 2: Worker/Model Runner Transfer Semantics
Tasks:
- Validate and patch no-forward transfer iteration (`kv_connector_no_forward`) behavior.
- Confirm connector output propagation in PP and non-PP paths.
- Confirm transfer-group registration and host copy ops are correct for RBLN KV layout.

Tests first (worker):
- New tests in `tests/v1/worker` for:
  - transfer-only step returns connector output correctly.
  - model runner returns empty/no-forward output only when appropriate.
  - connector send/recv completion flags are propagated.

Done when:
- Worker and model-runner KV transfer tests pass.

### Phase 3: LMCache Connector Integration
Tasks:
- Verify RBLN path works with `KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role=...)`.
- Add minimal RBLN docs/examples for LMCache producer/consumer usage.
- Validate required env handling (`LMCACHE_CONFIG_FILE`, connector extra config).
- Split validation into:
  - fallback mode (`LMCacheConnectorV1` routed to `RBLNHostConnector`) for immediate functionality.
  - strict mode (`--strict-lmcache`) for LMCache cache hit/load validation on
    shared backend with deterministic hash config.

Tests first:
- Add integration-focused tests where feasible with mocked connector interfaces.
- Add smoke scripts (manual execution) for shared-storage baseline and LMCache flow.

Done when:
- Shared-storage and LMCache smoke scripts produce real generation output in both phases.

Open items for strict LMCache completion:
- Upstream persistent LocalDisk index restoration support in LMCache so
  RBLN does not require runtime monkeypatching in connector code.
- Extend strict mode regression coverage for MLA and mixed backend modes
  beyond current non-MLA shared-local-disk smoke.
- Re-enable compile/warmup path once the `librbln` abort root cause is fixed.

### Phase 4: Hardening and Regression Coverage
Tasks:
- Add edge-case tests:
  - request preemption during KV transfer
  - connector failure path (`failed_recving_kv_req_ids`)
  - chunked prefill interaction
- Update docs and runbook.

Tests first:
- Extend unit tests for all above edge paths before code changes.

Done when:
- Core and worker suites are green and no disaggregation regressions are observed.

## File-Level Work Plan
- Scheduler:
  - `vllm_rbln/v1/core/rbln_scheduler.py`
  - `tests/v1/core/test_scheduler.py`
  - `tests/v1/core/test_async_scheduler.py` (if async paths are touched)
- Worker/Runner:
  - `vllm_rbln/v1/worker/rbln_worker.py`
  - `vllm_rbln/v1/worker/rbln_model_runner.py`
  - `tests/v1/worker/*` (new KV transfer coverage)
- Optimum guard:
  - `vllm_rbln/v1/core/optimum_scheduler.py`
  - `vllm_rbln/platform.py` (if centralized guard is cleaner)
- Examples:
  - `examples/experimental/disaggregation/run_disagg_shared_storage_smoke.py`
  - `examples/experimental/disaggregation/run_disagg_lmcache_smoke.py`
  - `examples/experimental/disaggregation/README.md`

## Validation Matrix
- Unit:
  - scheduler KV connector lifecycle
  - worker connector output semantics
- Integration:
  - shared-storage producer->consumer smoke
  - LMCache producer->consumer smoke
- Acceptance:
  - no hardcoded expected text
  - decode phase generates real output
  - connector metadata path exercised

## Anti-Patterns (Disallowed)
- Hardcoding generated text or token ids in smoke scripts to force pass.
- Marking connector tests as pass without executing connector paths.
- Silencing errors from connector init/recv and still returning success.

## Exit Criteria
- New connector tests fail before implementation and pass after implementation.
- RBLN v1 path supports prefill-decode disaggregation with LMCache connector config.
- Example scripts and README are usable as bring-up references.
- No regression in existing scheduler/worker test suites.
