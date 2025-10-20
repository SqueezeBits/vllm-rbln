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

import argparse
import dataclasses
import json
import math
import os
import time
from types import TracebackType
from typing import Any, Optional, Union

from vllm import LLM, EngineArgs

from vllm_rbln.rbln_envs import RBLN_USE_VLLM_MODEL


def get_llm_instance(engine_args: EngineArgs) -> LLM:
    if not RBLN_USE_VLLM_MODEL:
        compiled_model_path = engine_args.model
        if not os.path.exists(compiled_model_path):
            raise ValueError(
                f"Compiled model path does not exist: {compiled_model_path}")

    return LLM(**dataclasses.asdict(engine_args))


def convert_to_pytorch_benchmark_format(args: argparse.Namespace,
                                        metrics: dict[str, list],
                                        extra_info: dict[str, Any]) -> list:
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get(
            "tensor_parallel_size")
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"][
                "tensor_parallel_size"] = (extra_info["tensor_parallel_size"])

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):

    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(
            records,
            f,
            cls=InfEncoder,
            default=lambda o:
            f"<{type(o).__name__} object is not JSON serializable>",
        )


# Collect time and generate time metrics
#
# Example Usage:
#   collector = TimeCollector(TimeCollector.US)
#   for _ in range(total_iteration):
#      with collector:
#          ...
#   collector.dump_avg_max()
class TimeCollector:
    NS: int = 1
    US: int = NS * 1000
    MS: int = US * 1000
    S: int = MS * 1000

    def __init__(self, scale: int) -> None:
        self.cnt: int = 0
        self._sum: int = 0
        self._max: Optional[int] = None
        self.scale = scale
        self.start_time: int = time.monotonic_ns()

    def collect(self, v: int) -> None:
        self.cnt += 1
        self._sum += v
        if self._max is None:
            self._max = v
        else:
            self._max = max(self._max, v)

    def avg(self) -> Union[float, str]:
        return (self._sum * 1.0 / self.cnt /
                self.scale if self.cnt > 0 else "N/A")

    def max(self) -> Union[float, str]:
        return self._max / self.scale if self._max else "N/A"

    def dump_avg_max(self) -> list[Union[float, str]]:
        return [self.avg(), self.max()]

    def __enter__(self) -> None:
        self.start_time = time.monotonic_ns()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        self.collect(time.monotonic_ns() - self.start_time)
