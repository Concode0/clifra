import os
from dataclasses import replace

import pytest
import torch

from benchmarks.adapters import CLIFRA_ADAPTER
from benchmarks.cases import BenchmarkCase, ExecutionPlacement, LayoutCase, SelectionCase
from benchmarks.engine import MeasurementConfig, measure_request
from benchmarks.model import BenchmarkRequest, ConstructedBenchmark, PreparedBenchmark
from benchmarks.run import deterministic_order, run_cases, run_requests


def tiny_case():
    return BenchmarkCase(
        case_id="product.engine.tiny",
        operation="geometric_product",
        signature=(2, 0, 0),
        inputs=(LayoutCase((0, 1, 2), leading_shape=(2,)),) * 2,
        output=LayoutCase((0, 1, 2)),
    )


def measure_case(case, mode, config, *, placement=ExecutionPlacement()):
    return measure_request(BenchmarkRequest(case, placement, mode), CLIFRA_ADAPTER, config)


class MinimalAdapter:
    name = "minimal-test-backend"

    def preflight(self, request):
        return None

    def construct(self, request, *, requires_grad=False):
        value = torch.ones(1, requires_grad=requires_grad)
        return ConstructedBenchmark(None, (value,), (), None)

    def prepare(self, request, constructed):
        return PreparedBenchmark(lambda value: value * 2, constructed, {"custom": {"route": "test"}})

    def invoke(self, prepared, *, backward):
        result = prepared.operation(*prepared.constructed.arguments)
        if backward:
            result.square().sum().backward()

    def reset(self, prepared, *, backward):
        if backward:
            prepared.constructed.arguments[0].grad = None

    def synchronize(self, request):
        return None

    def synchronization_description(self, request):
        return "test backend synchronous"


class LifecycleAdapter(MinimalAdapter):
    def __init__(self):
        self.constructed_ids = []
        self.prepared_ids = []
        self.invoked_ids = []

    def construct(self, request, *, requires_grad=False):
        instance_id = len(self.constructed_ids) + 1
        self.constructed_ids.append(instance_id)
        value = torch.ones(1, requires_grad=requires_grad)
        contract = {"constructed_id": instance_id}
        return ConstructedBenchmark(instance_id, (value,), (contract,), contract)

    def prepare(self, request, constructed):
        preparation_id = len(self.prepared_ids) + 1
        self.prepared_ids.append((preparation_id, constructed.state))
        return PreparedBenchmark(
            lambda value: value * 2,
            constructed,
            {
                "preparation_id": preparation_id,
                "constructed_id": constructed.state,
            },
        )

    def invoke(self, prepared, *, backward):
        self.invoked_ids.append(prepared.metadata["preparation_id"])
        super().invoke(prepared, backward=backward)


@pytest.mark.parametrize(
    "mode,expected_warmups",
    [
        ("construction_setup", 1),
        ("planning_preparation", 1),
        ("first_invocation", 0),
        ("steady_forward", 1),
        ("forward_backward", 1),
    ],
)
def test_each_timing_stage_preserves_raw_samples(mode, expected_warmups):
    row = measure_case(tiny_case(), mode, MeasurementConfig(samples=2, warmups=1))

    assert row["status"] == "ok"
    assert row["timing"]["warmup_count"] == expected_warmups
    assert len(row["timing"]["samples_ns"]) == 2
    assert row["timing"]["summary_ns"]["min_ns"] == min(row["timing"]["samples_ns"])


def test_measurement_engine_accepts_adapter_owned_execution_and_metadata():
    request = BenchmarkRequest(tiny_case(), ExecutionPlacement(), "forward_backward")

    row = measure_request(request, MinimalAdapter(), MeasurementConfig(samples=2, warmups=1))

    assert row["status"] == "ok"
    assert row["backend"] == "minimal-test-backend"
    assert row["backend_metadata"] == {"custom": {"route": "test"}}
    assert row["timing"]["synchronization"] == "test backend synchronous"
    assert len(row["timing"]["samples_ns"]) == 2


def test_construction_stage_never_prepares_and_observes_a_measured_construction():
    adapter = LifecycleAdapter()
    request = BenchmarkRequest(tiny_case(), ExecutionPlacement(), "construction_setup")

    row = measure_request(request, adapter, MeasurementConfig(samples=2, warmups=1))

    assert adapter.constructed_ids == [1, 2, 3]
    assert adapter.prepared_ids == []
    assert adapter.invoked_ids == []
    assert row["contracts"] == {
        "inputs": [{"constructed_id": 3}],
        "output": {"constructed_id": 3},
    }
    assert row["backend_metadata"] is None


def test_planning_stage_observes_an_actually_measured_preparation():
    adapter = LifecycleAdapter()
    request = BenchmarkRequest(tiny_case(), ExecutionPlacement(), "planning_preparation")

    row = measure_request(request, adapter, MeasurementConfig(samples=2, warmups=1))

    assert adapter.constructed_ids == [1, 2, 3]
    assert adapter.prepared_ids == [(1, 1), (2, 2), (3, 3)]
    assert adapter.invoked_ids == []
    assert row["contracts"]["inputs"] == [{"constructed_id": 3}]
    assert row["backend_metadata"] == {"preparation_id": 3, "constructed_id": 3}


def test_first_invocation_metadata_comes_from_the_invoked_fresh_preparation():
    adapter = LifecycleAdapter()
    request = BenchmarkRequest(tiny_case(), ExecutionPlacement(), "first_invocation")

    row = measure_request(request, adapter, MeasurementConfig(samples=2, warmups=4))

    assert adapter.constructed_ids == [1, 2]
    assert adapter.prepared_ids == [(1, 1), (2, 2)]
    assert adapter.invoked_ids == [1, 2]
    assert row["timing"]["warmup_count"] == 0
    assert row["contracts"]["inputs"] == [{"constructed_id": 2}]
    assert row["backend_metadata"] == {"preparation_id": 2, "constructed_id": 2}


@pytest.mark.parametrize("mode", ["steady_forward", "forward_backward"])
def test_steady_stages_observe_the_single_preparation_they_measure(mode):
    adapter = LifecycleAdapter()
    request = BenchmarkRequest(tiny_case(), ExecutionPlacement(), mode)

    row = measure_request(request, adapter, MeasurementConfig(samples=2, warmups=1))

    assert adapter.constructed_ids == [1]
    assert adapter.prepared_ids == [(1, 1)]
    assert adapter.invoked_ids == [1, 1, 1]
    assert row["contracts"]["inputs"] == [{"constructed_id": 1}]
    assert row["backend_metadata"] == {"preparation_id": 1, "constructed_id": 1}


def test_execution_failure_produces_a_failed_untimed_row():
    class FailingAdapter(MinimalAdapter):
        def invoke(self, prepared, *, backward):
            raise RuntimeError("deliberate execution failure")

    request = BenchmarkRequest(tiny_case(), ExecutionPlacement(), "steady_forward")
    row = measure_request(request, FailingAdapter(), MeasurementConfig(samples=1, warmups=0))

    assert row["status"] == "failed"
    assert row["error"] == {"type": "RuntimeError", "message": "deliberate execution failure"}
    assert row["backend_metadata"] == {"custom": {"route": "test"}}
    assert row["timing"]["samples_ns"] == []
    assert row["timing"]["summary_ns"] is None


def test_unavailable_mps_is_explicitly_unsupported_and_untimed(monkeypatch):
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    row = measure_case(
        tiny_case(),
        "steady_forward",
        MeasurementConfig(samples=1, warmups=0),
        placement=ExecutionPlacement(device="mps"),
    )

    assert row["status"] == "unsupported"
    assert row["timing"]["samples_ns"] == []
    assert row["timing"]["summary_ns"] is None


def test_unavailable_cuda_is_explicitly_unsupported_and_untimed(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    row = measure_case(
        tiny_case(),
        "steady_forward",
        MeasurementConfig(samples=1, warmups=0),
        placement=ExecutionPlacement(device="cuda"),
    )

    assert row["status"] == "unsupported"
    assert row["error"]["type"] == "UnsupportedDevice"
    assert row["timing"]["samples_ns"] == []


def test_cuda_synchronization_is_centralized(monkeypatch):
    calls = []
    monkeypatch.setattr("torch.cuda.synchronize", lambda: calls.append("synchronize"))

    request = BenchmarkRequest(tiny_case(), ExecutionPlacement(device="cuda"), "steady_forward")
    CLIFRA_ADAPTER.synchronize(request)

    assert calls == ["synchronize"]


def test_forced_route_uses_only_repository_private_adapter():
    placement = ExecutionPlacement(selection=SelectionCase("forced_repository_private", "sparse"))
    row = measure_case(tiny_case(), "steady_forward", MeasurementConfig(samples=1, warmups=0), placement=placement)

    assert row["status"] == "ok"
    assert row["backend_metadata"]["selection_mode"] == "forced_repository_private"
    assert row["backend_metadata"]["selected_route"] == "sparse"


def test_ineligible_forced_route_is_unsupported_and_untimed():
    case = BenchmarkCase(
        case_id="product.engine.ineligible",
        operation="wedge",
        signature=(2, 0, 0),
        inputs=(LayoutCase((1,), leading_shape=(2,)),) * 2,
        output=LayoutCase((2,)),
    )
    placement = ExecutionPlacement(selection=SelectionCase("forced_repository_private", "full_table"))
    row = measure_case(case, "steady_forward", MeasurementConfig(samples=1, warmups=0), placement=placement)

    assert row["status"] == "unsupported"
    assert row["error"]["type"] == "IneligibleRoute"
    assert row["timing"]["samples_ns"] == []


def test_runner_measures_all_stages_in_the_current_process():
    artifact = run_cases(
        (tiny_case(),),
        ("steady_forward", "first_invocation"),
        MeasurementConfig(samples=1, warmups=0),
    )

    process_ids = {row["provenance"]["process_id"] for row in artifact["rows"]}
    assert process_ids == {os.getpid()}
    assert artifact["campaign"]["execution_strategy"] == "in_process"


def test_request_model_round_trips_and_drives_the_measurement_layer_directly():
    request = BenchmarkRequest(tiny_case(), ExecutionPlacement(), "steady_forward")

    row = measure_request(
        BenchmarkRequest.from_dict(request.to_dict()),
        CLIFRA_ADAPTER,
        MeasurementConfig(samples=1, warmups=0),
    )

    assert row["status"] == "ok"
    assert row["backend"] == "clifra"
    assert row["row_id"] == request.row_id


def test_row_failure_does_not_abort_the_rest_of_a_campaign():
    case = tiny_case()
    requests = (
        BenchmarkRequest(
            case,
            ExecutionPlacement(selection=SelectionCase("forced_repository_private", "missing")),
            "steady_forward",
        ),
        BenchmarkRequest(case, ExecutionPlacement(), "steady_forward"),
    )

    artifact = run_requests(
        requests,
        MeasurementConfig(samples=1, warmups=0),
    )

    assert {row["status"] for row in artifact["rows"]} == {"ok", "unsupported"}


def test_deterministic_shuffle_is_reproducible_and_input_order_independent():
    requests = tuple(
        BenchmarkRequest(tiny_case(), ExecutionPlacement(), mode)
        for mode in (
            "construction_setup",
            "planning_preparation",
            "first_invocation",
            "steady_forward",
            "forward_backward",
        )
    )

    assert deterministic_order(requests, 91) == deterministic_order(tuple(reversed(requests)), 91)
    assert deterministic_order(requests, 91) != deterministic_order(requests, 92)


def test_checkpoint_resume_reuses_identical_completed_rows(tmp_path):
    request = BenchmarkRequest(tiny_case(), ExecutionPlacement(), "steady_forward")
    checkpoint = tmp_path / "campaign.jsonl"
    config = MeasurementConfig(samples=1, warmups=0)

    first = run_requests((request,), config, checkpoint_path=checkpoint)
    second = run_requests((request,), config, checkpoint_path=checkpoint)

    assert first["campaign"]["checkpoint_rows_reused"] == 0
    assert second["campaign"]["checkpoint_rows_reused"] == 1
    assert second["rows"] == first["rows"]
    assert len(checkpoint.read_text().splitlines()) == 1


def test_campaign_checkpoints_each_row_before_a_later_interruption(tmp_path, monkeypatch):
    requests = tuple(
        BenchmarkRequest(
            replace(tiny_case(), case_id=f"product.engine.interrupt-{index}"), ExecutionPlacement(), "steady_forward"
        )
        for index in range(2)
    )
    checkpoint = tmp_path / "interrupted.jsonl"
    config = MeasurementConfig(samples=1, warmups=0)

    from benchmarks import run as benchmark_run

    original_measure = benchmark_run.measure_request
    calls = 0

    def interrupt_second_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated interruption")
        return original_measure(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr("benchmarks.run.measure_request", interrupt_second_request)
        with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
            run_requests(requests, config, checkpoint_path=checkpoint)

    assert len(checkpoint.read_text().splitlines()) == 1
    resumed = run_requests(requests, config, checkpoint_path=checkpoint)
    assert resumed["campaign"]["checkpoint_rows_reused"] == 1
    assert len(resumed["rows"]) == 2
