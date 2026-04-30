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

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

# ---------------------------------------------------------------------------
# Local prefill scheduling restriction
# ---------------------------------------------------------------------------


def is_prefill_request(request: Request) -> bool:
    """Return whether ``request`` is a prefill request."""
    return request.num_computed_tokens < request.num_tokens - 1


def get_running_schedule_start_index(running: list[Request]) -> int:
    """Return where RBLN should start scanning the running queue.

    Upstream scans running requests from index 0. RBLN cannot mix local prefill
    with decode, and local prefill batch size is currently 1. When a prefill
    request is already running, it is kept at the tail of the running queue, so
    starting from the tail lets the scheduler continue that prefill before
    scheduling earlier decode requests.
    """
    if running and is_prefill_request(running[-1]):
        return len(running) - 1
    return 0


# ---------------------------------------------------------------------------
# Decode scheduling restriction
# ---------------------------------------------------------------------------


def get_decode_batch_cap(
    max_num_running_reqs: int,
    pipeline_parallel_size: int,
) -> int:
    """Return the per-step decode batch cap for RBLN pipeline parallelism.

    Upstream limits the total running queue with ``max_num_running_reqs``.
    RBLN additionally limits the per-step decode batch to
    ``max_num_running_reqs // pipeline_parallel_size`` to prevent pipeline
    bubbles. This matches the decode batch buckets compiled by the RBLN runner,
    which are sized per pipeline stage rather than for the full running queue.
    """
    return max_num_running_reqs // pipeline_parallel_size


# ---------------------------------------------------------------------------
# Spec decode boundary restriction
# ---------------------------------------------------------------------------


def update_spec_decode_cap(
    request: Request,
    current_cap: int,
    block_size: int,
    max_model_len: int,
) -> int:
    """Tighten the batch-wide spec decode cap using a scheduled request.

    RBLN pads all requests in a decode batch to the maximum scheduled token
    length. For decode requests, the cap must respect both the remaining space
    in the current KV block and the remaining positions before max model length.
    This is called only after the request has been accepted for scheduling, so
    the request should affect the cap only when it will actually be emitted in
    this step. Even single-token decode requests constrain the cap because they
    can be padded up to the batch max spec decode length by the runner. Prefill
    requests do not constrain speculative decode.
    """
    if is_prefill_request(request):
        return current_cap

    tokens_used_in_block = request.num_computed_tokens % block_size
    remaining_in_block = block_size - tokens_used_in_block
    remaining_in_maxlen = max_model_len - request.num_computed_tokens
    return min(remaining_in_block, remaining_in_maxlen, current_cap)


def trim_scheduled_tokens_to_spec_decode_cap(
    scheduled_running_reqs: list[Request],
    num_scheduled_tokens: dict[str, int],
    scheduled_spec_decode_tokens: dict[str, list[int]],
    spec_decode_cap: int,
) -> int:
    """Apply the final spec decode cap to already scheduled running requests.

    A later scheduled request can tighten the batch-wide cap after earlier
    requests already received more speculative tokens. Trim those earlier
    entries in-place and return the number of tokens reclaimed for the scheduler
    token budget.

    Precondition: ``scheduled_spec_decode_tokens`` is non-empty.
    This is only needed when spec decode tokens were scheduled in this step. In
    that case, the runner pads all decode requests to the batch max spec decode
    length. Without this retroactive trim, a request near a KV block boundary
    could be padded beyond its allowed boundary even if its own scheduled token
    count was already capped. The spec token list is re-trimmed to keep it
    consistent with the reduced ``num_scheduled_tokens`` value.
    """
    reclaimed_tokens = 0

    for request in scheduled_running_reqs:
        request_id = request.request_id
        old_num_scheduled_tokens = num_scheduled_tokens[request_id]
        if old_num_scheduled_tokens <= spec_decode_cap:
            continue

        new_num_scheduled_tokens = spec_decode_cap
        reclaimed_tokens += old_num_scheduled_tokens - new_num_scheduled_tokens
        num_scheduled_tokens[request_id] = new_num_scheduled_tokens

        num_spec_tokens = (
            new_num_scheduled_tokens
            + request.num_computed_tokens
            - request.num_tokens
            - request.num_output_placeholders
        )
        if num_spec_tokens > 0:
            if request_id in scheduled_spec_decode_tokens:
                scheduled_spec_decode_tokens[request_id] = scheduled_spec_decode_tokens[
                    request_id
                ][:num_spec_tokens]
        else:
            scheduled_spec_decode_tokens.pop(request_id, None)

    return reclaimed_tokens


# ---------------------------------------------------------------------------
# No mixed batching / prefill policy
# ---------------------------------------------------------------------------


def remove_scheduled_running_reqs_for_prefill(
    scheduled_running_reqs: list[Request],
    req_to_new_blocks: dict[str, KVCacheBlocks],
    num_scheduled_tokens: dict[str, int],
    scheduled_spec_decode_tokens: dict[str, list[int]],
    scheduled_encoder_inputs: dict[str, list[int]],
) -> None:
    """Remove scheduled decode requests when a local prefill is selected.

    RBLN does not support mixed prefill/decode batches. If a local prefill is
    scheduled after running decode requests were tentatively selected, those
    decode requests are removed from this scheduler output. Any scheduled spec
    tokens are restored to the request so they can be considered again later.
    """
    for request in scheduled_running_reqs:
        request_id = request.request_id
        req_to_new_blocks.pop(request_id)
        num_scheduled_tokens.pop(request_id)
        request.spec_token_ids = scheduled_spec_decode_tokens.pop(request_id, [])
        scheduled_encoder_inputs.pop(request_id, None)

    scheduled_running_reqs.clear()
