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


from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner
from vllm_rbln.v1.worker.rbln_worker import RBLNWorker


def test_get_kv_connector_handshake_metadata_returns_none_without_group():
    worker = RBLNWorker.__new__(RBLNWorker)
    with patch(
        "vllm_rbln.v1.worker.rbln_worker.has_kv_transfer_group",
        return_value=False,
    ):
        assert worker.get_kv_connector_handshake_metadata() is None


def test_get_kv_connector_handshake_metadata_returns_tp_ranked_payload():
    worker = RBLNWorker.__new__(RBLNWorker)
    payload = {"transport": "dummy"}

    with (
        patch(
            "vllm_rbln.v1.worker.rbln_worker.has_kv_transfer_group",
            return_value=True,
        ),
        patch(
            "vllm_rbln.v1.worker.rbln_worker.get_kv_transfer_group",
            return_value=SimpleNamespace(get_handshake_metadata=lambda: payload),
        ),
        patch(
            "vllm_rbln.v1.worker.rbln_worker.get_tp_group",
            return_value=SimpleNamespace(rank_in_group=3),
        ),
    ):
        assert worker.get_kv_connector_handshake_metadata() == {3: payload}


def _make_model_runner_for_no_forward():
    runner = RBLNModelRunner.__new__(RBLNModelRunner)
    runner.execute_model_state = None
    runner.cache_config = SimpleNamespace(kv_sharing_fast_prefill=False)
    runner.vllm_config = SimpleNamespace()
    runner._update_states = lambda _scheduler_output: None
    return runner


def test_model_runner_returns_empty_output_without_connector_metadata():
    runner = _make_model_runner_for_no_forward()
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=0,
        kv_connector_metadata=None,
    )

    no_forward_called = False

    def _no_forward(_scheduler_output, _vllm_config):
        nonlocal no_forward_called
        no_forward_called = True
        return "UNREACHABLE"

    runner.kv_connector_no_forward = _no_forward

    with patch(
        "vllm_rbln.v1.worker.rbln_model_runner.has_kv_transfer_group",
        return_value=True,
    ):
        output = runner.execute_model(scheduler_output)

    assert output is EMPTY_MODEL_RUNNER_OUTPUT
    assert not no_forward_called


def test_model_runner_calls_no_forward_when_connector_metadata_exists():
    runner = _make_model_runner_for_no_forward()
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=0,
        kv_connector_metadata={"dummy": "metadata"},
    )

    expected_output = object()
    no_forward_called = False

    def _no_forward(_scheduler_output, _vllm_config):
        nonlocal no_forward_called
        no_forward_called = True
        return expected_output

    runner.kv_connector_no_forward = _no_forward

    with patch(
        "vllm_rbln.v1.worker.rbln_model_runner.has_kv_transfer_group",
        return_value=True,
    ):
        output = runner.execute_model(scheduler_output)

    assert output is expected_output
    assert no_forward_called


class _FakeIntermediateTensors:
    def __init__(self, tensors, kv_connector_output=None):
        self.tensors = tensors
        self.kv_connector_output = kv_connector_output


class _FakePPGroup:
    def __init__(self, is_first_rank=True, is_last_rank=False):
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank
        self.sent_tensors = None

    def recv_tensor_dict(self):
        return {"hidden": 1}

    def send_tensor_dict(self, tensors):
        self.sent_tensors = tensors


class _FakeModelRunner:
    def __init__(self, output):
        self._output = output
        self.intermediate_inputs = []

    def execute_model(self, scheduler_output, intermediate_tensors):
        self.intermediate_inputs.append(intermediate_tensors)
        return self._output


def _make_worker_for_pp_path(model_runner):
    worker = RBLNWorker.__new__(RBLNWorker)
    worker.model_runner = model_runner
    worker.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(distributed_executor_backend="ray"),
    )
    return worker


def test_worker_pp_path_returns_empty_output_for_empty_connector_state():
    kv_output = KVConnectorOutput()
    intermediate = _FakeIntermediateTensors(
        tensors={"hidden": 7},
        kv_connector_output=kv_output,
    )
    worker = _make_worker_for_pp_path(_FakeModelRunner(intermediate))
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    pp_group = _FakePPGroup(is_first_rank=True, is_last_rank=False)

    with (
        patch(
            "vllm_rbln.v1.worker.rbln_worker.IntermediateTensors",
            _FakeIntermediateTensors,
        ),
        patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group", return_value=pp_group),
    ):
        output = worker.execute_model(scheduler_output)

    assert output is EMPTY_MODEL_RUNNER_OUTPUT
    assert pp_group.sent_tensors == {"hidden": 7}


def test_worker_pp_path_propagates_finished_send_recv_flags():
    kv_output = KVConnectorOutput(
        finished_sending={"req-send"},
        finished_recving={"req-recv"},
    )
    intermediate = _FakeIntermediateTensors(
        tensors={"hidden": 9},
        kv_connector_output=kv_output,
    )
    worker = _make_worker_for_pp_path(_FakeModelRunner(intermediate))
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    pp_group = _FakePPGroup(is_first_rank=True, is_last_rank=False)

    with (
        patch(
            "vllm_rbln.v1.worker.rbln_worker.IntermediateTensors",
            _FakeIntermediateTensors,
        ),
        patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group", return_value=pp_group),
    ):
        output = worker.execute_model(scheduler_output)

    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert output.kv_connector_output is kv_output
    assert output.kv_connector_output.finished_sending == {"req-send"}
    assert output.kv_connector_output.finished_recving == {"req-recv"}


def test_worker_pp_path_propagates_invalid_block_ids_without_finished_flags():
    kv_output = KVConnectorOutput(invalid_block_ids={101, 202})
    intermediate = _FakeIntermediateTensors(
        tensors={"hidden": 11},
        kv_connector_output=kv_output,
    )
    worker = _make_worker_for_pp_path(_FakeModelRunner(intermediate))
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    pp_group = _FakePPGroup(is_first_rank=True, is_last_rank=False)

    with (
        patch(
            "vllm_rbln.v1.worker.rbln_worker.IntermediateTensors",
            _FakeIntermediateTensors,
        ),
        patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group", return_value=pp_group),
    ):
        output = worker.execute_model(scheduler_output)

    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert output.kv_connector_output is kv_output
    assert output.kv_connector_output.invalid_block_ids == {101, 202}
