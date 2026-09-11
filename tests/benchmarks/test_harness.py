"""Protect semantic identity, strict forcing, gates, and measurement lifecycle."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmarks.adapter import make_inputs, prepare, temporary_workarounds
from benchmarks.cases import TRACKS, core_cases, sweep
from benchmarks.correctness import check, gate
from benchmarks.run import availability, observations
from benchmarks.timing import measure, workload
from clifra.core import AlgebraContext
from clifra.core._kernel.planning.policy import NoAvailableRouteError
from clifra.core._kernel.routing import ExecutorRouter


def test_catalog_and_sweeps_have_stable_unique_ids():
    cases = core_cases()
    assert len(cases) == len({c.id for c in cases}) == 28
    for track in TRACKS:
        generated = list(sweep(track, (2, 3)))
        assert len(generated) == len({c.id for c in generated})
        assert generated == list(sweep(track, (2, 3)))
        assert all(c.reason for c in generated)


@pytest.mark.parametrize("route", ["vector_matrix", "rotor_product", "full_action_matrix"])
@pytest.mark.parametrize("storage", ["compact", "canonical"])
def test_reflection_track_matches_independent_output_and_vjp(route, storage):
    case = next(sweep("action-reflections", (3,)))
    a = AlgebraContext(*case.signature, dtype=torch.float64)
    args, _ = make_inputs(a, case, storage, 1024**2)
    if route != "vector_matrix":
        with pytest.raises(NoAvailableRouteError, match="unsupported_action_domain"):
            prepare(a, case, route, storage)
        return
    op, metadata, _ = prepare(a, case, route, storage)
    gate(a, case, storage, op, args, True)
    assert not temporary_workarounds(case, metadata, tuple(x[:1] for x in args))


@pytest.mark.parametrize(
    "operation",
    ["signature_norm_squared", "lane_grade_energy", "conjugate_scalar_form", "reverse", "pseudoscalar_product"],
)
def test_forms_track_matches_independent_output_and_vjp(operation):
    case = next(c for c in sweep("forms", (3,)) if c.operation == operation)
    a = AlgebraContext(*case.signature, dtype=torch.float64)
    args, _ = make_inputs(a, case, "compact", 1024**2)
    op, _, _ = prepare(a, case, "default", "compact")
    gate(a, case, "compact", op, args, True)


def test_full_layout_reflection_admits_materialized_route():
    case = next(c for c in sweep("action-reflections", (3,)) if len(c.output) == 4)
    a = AlgebraContext(*case.signature, dtype=torch.float64)
    args, _ = make_inputs(a, case, "compact", 1024**2)
    op, metadata, _ = prepare(a, case, "full_action_matrix", "compact")
    gate(a, case, "compact", op, args, True)
    assert not temporary_workarounds(case, metadata, args)


@pytest.mark.parametrize("storage", ["compact", "canonical"])
@pytest.mark.parametrize(
    "id,route",
    [
        ("product/gp/e3/full/single", "sparse"),
        ("product/gp/e3/full/single", "full_table"),
        ("product/gp/e5/vector/pairwise", "sparse"),
        ("exp/e3/simple/l1-0p25/b32", "taylor"),
        ("action/e3/vector/shared-bivector", "rotor_product"),
        ("grade/e8/project/012-to-1/b256", "default"),
    ],
)
def test_prepared_routes_preserve_output_and_vjp_without_replanning(storage, id, route, monkeypatch):
    case = next(c for c in core_cases() if c.id == id)
    a = AlgebraContext(*case.signature, dtype=torch.float64)
    args, seed = make_inputs(a, case, storage, 256 * 1024**2)
    again, again_seed = make_inputs(a, case, storage, 256 * 1024**2)
    assert seed == again_seed
    for x, y in zip(args, again):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    op, metadata, _ = prepare(a, case, route, storage)
    if route != "default":
        assert metadata["route"] == route
    result, oracle = gate(a, case, storage, op, args, True)
    assert result["status"] == "passed"

    def fail(*args, **kwargs):
        raise AssertionError("execution reentered planning")

    monkeypatch.setattr(ExecutorRouter, "select", fail)
    check(op, oracle, args, True, torch.float64)


def test_forcing_never_silently_falls_back_or_reinterprets_canonical_storage():
    case = next(c for c in core_cases() if c.id == "product/gp/e16/vector/b64")
    a = AlgebraContext(*case.signature)
    with pytest.raises(NoAvailableRouteError, match="requires_canonical_full_layouts"):
        prepare(a, case, "full_table", "canonical")


def test_gate_rejects_wrong_values_and_gradients():
    x = torch.tensor([1.0, 2.0])
    with pytest.raises(AssertionError):
        check(lambda x: x + 1, lambda x: x, (x,), False, x.dtype)
    with pytest.raises(AssertionError):
        check(lambda x: x * 2 - x.detach(), lambda x: x, (x,), True, x.dtype)


def test_backward_uses_new_graph_and_clears_leaf_gradients():
    retained = []

    def fn(x):
        retained.append(x)
        return x.square()

    step = workload(fn, (torch.tensor([2.0]),), True)
    result = measure(step, "cpu", warmup=2, samples=3, repetitions=2)
    assert retained[-1].grad.item() == 2.0
    assert result["count"] == 3
    assert len(result["samples_ms"]) == 3
    assert result["min_ms"] <= result["median_ms"]


def test_zero_interaction_product_has_zero_vjp():
    case = replace(core_cases()[0], inputs=((0,), (0,)), output=(1,))
    a = AlgebraContext(*case.signature)
    args, _ = make_inputs(a, case, "compact", 1024)
    op, _, _ = prepare(a, case, "sparse", "compact")
    gate(a, case, "compact", op, args, True)


def test_mps_float64_has_explicit_rejection():
    assert availability("mps", "float64") in {"mps_device_unavailable", "mps_does_not_support_float64"}


def test_observations_require_separated_distributions():
    def row(route, median, low, high):
        return {
            "semantic_case_id": "case",
            "status": "ok",
            "requested_route": route,
            "policy_selected_route": "sparse",
            "implementation": {"route": route},
            "timings": {"steady": {"median_ms": median, "p10_ms": low, "p90_ms": high}},
        }

    assert observations([row("default", 2.0, 1.0, 3.0), row("full_table", 1.0, 0.5, 2.0)]) == []
    assert len(observations([row("default", 2.0, 1.8, 2.2), row("full_table", 1.0, 0.9, 1.1)])) == 1
    affected = row("default", 2.0, 1.8, 2.2)
    affected["temporary_workarounds"] = ["pytorch_singleton_matrix_exp_batch_duplication"]
    affected["policy_comparison_eligible"] = False
    assert observations([affected, row("full_table", 1.0, 0.9, 1.1)]) == []
    alternative = row("full_table", 1.0, 0.9, 1.1)
    alternative.update(temporary_workarounds=affected["temporary_workarounds"], policy_comparison_eligible=False)
    assert observations([row("default", 2.0, 1.8, 2.2), alternative]) == []


@pytest.mark.parametrize("leading,affected", [((), True), ((1,), True), ((1, 1), True), ((2,), False), ((2, 1), False)])
def test_workaround_metadata_matches_parameter_batch(leading, affected):
    case = next(c for c in core_cases() if c.id == "action/e3/vector/shared-bivector")
    a = AlgebraContext(3)
    args = (torch.zeros(3), torch.zeros(*leading, 3))
    _, metadata, _ = prepare(a, case, "vector_matrix", "compact")
    assert bool(temporary_workarounds(case, metadata, args)) == affected
    _, metadata, _ = prepare(a, case, "rotor_product", "compact")
    assert temporary_workarounds(case, metadata, args) == []


def test_workaround_metadata_includes_materialized_exp_and_action_children():
    exp = next(c for c in core_cases() if c.family == "bivector_exp")
    a = AlgebraContext(*exp.signature)
    _, metadata, _ = prepare(a, exp, "left_matrix_exp", "compact")
    assert temporary_workarounds(exp, metadata, (torch.zeros(3),))
    assert not temporary_workarounds(exp, metadata, (torch.zeros(2, 3),))
    action = next(c for c in core_cases() if c.family == "action")
    from tests.helpers.policy import PreferRoute

    a._planner.policy = PreferRoute("bivector_exp", "left_matrix_exp")
    _, metadata, _ = prepare(a, action, "rotor_product", "compact")
    assert temporary_workarounds(action, metadata, (torch.zeros(128, 3), torch.zeros(3)))


def test_failed_gate_never_enters_timing(monkeypatch):
    import benchmarks.run as runner

    def reject(*args, **kwargs):
        raise AssertionError("deliberate incorrect implementation")

    def forbidden(*args, **kwargs):
        pytest.fail("timing reached after failed correctness")

    monkeypatch.setattr(runner, "gate", reject)
    monkeypatch.setattr(runner, "measure", forbidden)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    options = SimpleNamespace(
        storage="compact",
        device="cpu",
        dtype="float32",
        mode="eager",
        threads=1,
        reference_terms=200_000,
        reference_order=128,
        max_bytes=1024**2,
    )
    result = runner.execute(core_cases()[0], "default", options)
    assert result["status"] == "correctness_failed"
    assert "steady" not in result["timings"]


def test_large_layout_is_rejected_before_basis_enumeration():
    from benchmarks.adapter import Unsupported
    from benchmarks.correctness import reference

    case = replace(core_cases()[0], signature=(63, 0, 0), inputs=((31,), (31,)), output=(0,))
    with pytest.raises(Unsupported, match="max_lanes"):
        reference(AlgebraContext(63), case, "compact")
    action = replace(case, family="action", operation="versor", inputs=((1,), (2,)), output=(1,))
    with pytest.raises(Unsupported, match="matrix order"):
        reference(AlgebraContext(63), action, "compact")
