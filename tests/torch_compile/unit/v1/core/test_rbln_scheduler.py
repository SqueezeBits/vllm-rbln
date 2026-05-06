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

from vllm_rbln.v1.core.rbln_scheduler import RBLNScheduler, _is_prefill_request

from .utils import create_requests


def _request(num_computed_tokens: int, num_tokens: int):
    request = create_requests(num_requests=1, num_tokens=num_tokens)[0]
    request.num_computed_tokens = num_computed_tokens
    return request


def test_is_prefill_request_until_last_prompt_token() -> None:
    assert _is_prefill_request(_request(num_computed_tokens=0, num_tokens=8))
    assert _is_prefill_request(_request(num_computed_tokens=6, num_tokens=8))
    assert not _is_prefill_request(_request(num_computed_tokens=7, num_tokens=8))


def test_running_schedule_starts_at_tail_prefill() -> None:
    scheduler = object.__new__(RBLNScheduler)
    scheduler.running = [
        _request(num_computed_tokens=7, num_tokens=8),
        _request(num_computed_tokens=3, num_tokens=8),
    ]

    assert scheduler._get_running_schedule_start_index() == 1


def test_running_schedule_starts_at_zero_without_tail_prefill() -> None:
    scheduler = object.__new__(RBLNScheduler)
    scheduler.running = [
        _request(num_computed_tokens=3, num_tokens=8),
        _request(num_computed_tokens=7, num_tokens=8),
    ]

    assert scheduler._get_running_schedule_start_index() == 0
    scheduler.running = []
    assert scheduler._get_running_schedule_start_index() == 0


def test_prefill_request_does_not_update_spec_decode_cap() -> None:
    scheduler = object.__new__(RBLNScheduler)
    scheduler.block_size = 8
    scheduler.max_model_len = 32
    request = _request(num_computed_tokens=6, num_tokens=8)

    assert scheduler._update_spec_decode_cap(request, current_cap=8) == 8


def test_decode_request_updates_spec_decode_cap_to_block_boundary() -> None:
    scheduler = object.__new__(RBLNScheduler)
    scheduler.block_size = 8
    scheduler.max_model_len = 32
    request = _request(num_computed_tokens=7, num_tokens=8)

    assert scheduler._update_spec_decode_cap(request, current_cap=8) == 1


def test_decode_request_updates_spec_decode_cap_to_max_model_len() -> None:
    scheduler = object.__new__(RBLNScheduler)
    scheduler.block_size = 8
    scheduler.max_model_len = 7
    request = _request(num_computed_tokens=5, num_tokens=6)

    assert scheduler._update_spec_decode_cap(request, current_cap=8) == 2


def test_trim_scheduled_tokens_to_spec_decode_cap() -> None:
    request = _request(num_computed_tokens=8, num_tokens=8)
    request.spec_token_ids = []
    request.num_output_placeholders = 0
    num_scheduled_tokens = {request.request_id: 5}
    scheduled_spec_decode_tokens = {request.request_id: [1, 2, 3, 4]}

    reclaimed = RBLNScheduler._trim_scheduled_tokens_to_spec_decode_cap(
        scheduled_running_reqs=[request],
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        spec_decode_cap=3,
    )

    assert reclaimed == 2
    assert num_scheduled_tokens[request.request_id] == 3
    assert scheduled_spec_decode_tokens[request.request_id] == [1, 2, 3]


def test_trim_scheduled_tokens_removes_empty_spec_decode_tokens() -> None:
    request = _request(num_computed_tokens=7, num_tokens=8)
    request.spec_token_ids = []
    request.num_output_placeholders = 0
    num_scheduled_tokens = {request.request_id: 5}
    scheduled_spec_decode_tokens = {request.request_id: [1, 2, 3, 4]}

    reclaimed = RBLNScheduler._trim_scheduled_tokens_to_spec_decode_cap(
        scheduled_running_reqs=[request],
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        spec_decode_cap=1,
    )

    assert reclaimed == 4
    assert num_scheduled_tokens[request.request_id] == 1
    assert request.request_id not in scheduled_spec_decode_tokens


def test_remove_scheduled_running_reqs_for_prefill() -> None:
    req_a, req_b = create_requests(num_requests=2, num_tokens=8)
    req_a.num_computed_tokens = 8
    req_b.num_computed_tokens = 8
    scheduled_running_reqs = [req_a, req_b]
    req_to_new_blocks = {
        req_a.request_id: KVCacheBlocks(blocks=()),
        req_b.request_id: KVCacheBlocks(blocks=()),
    }
    num_scheduled_tokens = {
        req_a.request_id: 5,
        req_b.request_id: 3,
    }
    scheduled_spec_decode_tokens = {
        req_a.request_id: [1, 2, 3, 4],
    }
    scheduled_encoder_inputs = {
        req_b.request_id: [0],
    }

    RBLNScheduler._remove_scheduled_running_reqs_for_prefill(
        scheduled_running_reqs=scheduled_running_reqs,
        req_to_new_blocks=req_to_new_blocks,
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
    )

    assert scheduled_running_reqs == []
    assert req_to_new_blocks == {}
    assert num_scheduled_tokens == {}
    assert scheduled_spec_decode_tokens == {}
    assert scheduled_encoder_inputs == {}
    assert req_a.spec_token_ids == [1, 2, 3, 4]
    assert req_b.spec_token_ids == []
