import pytest

from benchmarks.cases import (
    DENSE_BATCHES,
    BenchmarkCase,
    ExecutionPlacement,
    LayoutCase,
    SelectionCase,
    exploratory_action_cases,
    exploratory_bivector_exp_cases,
    exploratory_product_cases,
    full_cases,
)
from benchmarks.run import _parse_args, campaign_plan


def test_case_and_placement_round_trip_separately():
    case = BenchmarkCase(
        case_id="product.round-trip",
        operation="wedge",
        signature=(2, 1, 0),
        inputs=(LayoutCase((1,), "canonical", (2, 1)), LayoutCase((1,), "compact", (1, 3))),
        output=LayoutCase((2,), "canonical"),
    )
    placement = ExecutionPlacement("float64", "cpu", SelectionCase("forced_repository_private", "sparse"))

    assert BenchmarkCase.from_dict(case.to_dict()) == case
    assert ExecutionPlacement.from_dict(placement.to_dict()) == placement
    assert "dtype" not in case.to_dict()
    assert "selection" not in case.to_dict()


@pytest.mark.parametrize("shape", [(0,), (-1,), (True,)])
def test_case_rejects_nonordinary_leading_dimensions(shape):
    with pytest.raises(ValueError, match="leading dimensions"):
        LayoutCase((1,), leading_shape=shape)


def test_product_matrix_preserves_existing_dense_axis_and_structures():
    cases = exploratory_product_cases()
    full_scaling = [case for case in cases if case.case_id.startswith("product.cl5.full-full.gp.batch")]

    assert tuple(case.inputs[0].leading_shape[0] for case in full_scaling) == DENSE_BATCHES
    assert {case.operation for case in cases} == {
        "geometric_product",
        "wedge",
        "left_contraction",
        "right_contraction",
        "symmetric_product",
        "commutator_product",
        "anti_commutator_product",
    }
    assert {sum(case.signature) for case in cases}.issuperset({3, 4, 5, 6, 8, 10, 12, 24})
    assert any(case.inputs[0].storage == "canonical" for case in cases)
    assert any(case.inputs[0].leading_shape != case.inputs[1].leading_shape for case in cases)


def test_exp_matrix_is_generated_from_domain_output_batch_and_structure_tracks():
    cases = exploratory_bivector_exp_cases()

    assert {sum(case.signature) for case in cases} == {2, 3, 4, 5, 6, 8, 10, 12}
    assert {case.generator_structure for case in cases} == {
        "random",
        "simple_plane",
        "commuting_planes",
        "null_plane",
    }
    assert {case.inputs[0].leading_shape for case in cases} == {(1,), (32,), (512,)}
    assert any(case.output.grades == (0,) for case in cases)
    assert any(case.output.grades == tuple(range(sum(case.signature) + 1)) for case in cases)
    assert len({case.case_id for case in cases}) == len(cases)


def test_action_matrix_covers_routed_request_and_broadcast_structures():
    cases = exploratory_action_cases()

    assert {case.operation for case in cases} == {"linear", "versor"}
    assert {case.action_grade for case in cases if case.operation == "versor"} == {1, 2}
    assert {sum(case.signature) for case in cases} == {3, 4, 5, 6, 8, 10, 12}
    assert any(
        case.inputs[0].leading_shape == (512,) and case.inputs[1].leading_shape == ()
        for case in cases
        if case.operation == "versor"
    )
    assert any(case.inputs[0].leading_shape == (32, 1) for case in cases)
    assert any(case.inputs[0].grades == tuple(range(sum(case.signature) + 1)) for case in cases)
    assert len({case.case_id for case in cases}) == len(cases)


def test_campaign_includes_normal_and_every_feasible_product_root_route():
    requests = campaign_plan(("cpu",), families=("product",), modes=("steady_forward",)).requests
    full = [
        request
        for request in requests
        if request.case.case_id == "product.cl5.full-full.gp.batch64" and request.placement.dtype == "float32"
    ]
    compact = [
        request
        for request in requests
        if request.case.case_id == "product.cl12.vector-bivector.gp.batch64" and request.placement.dtype == "float32"
    ]

    assert {(item.placement.selection.mode, item.placement.selection.route) for item in full} == {
        ("default", None),
        ("forced_repository_private", "full_table"),
        ("forced_repository_private", "sparse"),
    }
    assert {(item.placement.selection.mode, item.placement.selection.route) for item in compact} == {
        ("default", None),
        ("forced_repository_private", "sparse"),
    }


def test_full_cases_is_the_canonical_complete_suite():
    assert full_cases() == exploratory_product_cases() + exploratory_bivector_exp_cases() + exploratory_action_cases()


def test_campaign_can_pin_one_explicit_dtype():
    plan = campaign_plan(("cpu",), dtypes=("float32",), modes=("steady_forward",))
    assert {request.placement.dtype for request in plan.requests} == {"float32"}


def test_run_cli_defaults_to_full_single_explicit_placement():
    args = _parse_args(["--device", "mps", "--dtype", "float32", "--output", "artifact.json"])
    assert args.suite == "full"
    assert args.device == "mps"
    assert args.dtype == "float32"


def test_campaign_summary_reports_projected_rows_and_explicit_pruning():
    plan = campaign_plan(("cpu",), modes=("steady_forward",))
    summary = plan.summary()

    assert summary["semantic_case_count"] == len(plan.semantic_cases)
    assert summary["execution_row_count"] == len(plan.requests)
    assert summary["pruned_count"] == len(plan.pruned)
    assert {request.case.family for request in plan.requests} == {"product", "bivector_exp", "action"}
    assert {item["category"] for item in plan.pruned} <= {"capability_or_resource_limit"}
    assert any("max_pairs" in str(item["reason"]) for item in plan.pruned)
