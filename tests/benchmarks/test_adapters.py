from benchmarks.adapters import construct_case, plan_case, planning_metadata, route_assessments
from benchmarks.cases import ExecutionPlacement, SelectionCase, smoke_cases


def _case(family, operation=None):
    return next(
        case for case in smoke_cases() if case.family == family and (operation is None or case.operation == operation)
    )


def test_registered_root_routes_are_enumerated_per_exact_family_request():
    assert {item["route"] for item in route_assessments(_case("product"), ExecutionPlacement())} == {
        "full_table",
        "sparse",
    }
    assert {item["route"] for item in route_assessments(_case("bivector_exp"), ExecutionPlacement())} == {
        "closed",
        "taylor",
        "left_matrix_exp",
    }
    assert {item["route"] for item in route_assessments(_case("action", "versor"), ExecutionPlacement())} == {
        "vector_matrix",
        "rotor_product",
        "full_action_matrix",
        "graded_linear",
    }


def test_forced_action_root_leaves_exp_and_product_children_on_default_policy():
    case = _case("action", "versor")
    placement = ExecutionPlacement(selection=SelectionCase("forced_repository_private", "rotor_product"))
    algebra, inputs, output, _ = construct_case(case, placement)
    metadata = planning_metadata(plan_case(algebra, inputs, output, case), case, placement)

    assert metadata["selected_route"] == "rotor_product"
    assert {(item["family"], item["selected_route"]) for item in metadata["child_selections"]}.issuperset(
        {("bivector_exp", "closed"), ("product", "sparse")}
    )


def test_action_metadata_uses_route_level_exterior_facts():
    case = _case("action", "versor")
    placement = ExecutionPlacement(selection=SelectionCase("forced_repository_private", "vector_matrix"))
    algebra, inputs, output, _ = construct_case(case, placement)
    metadata = planning_metadata(plan_case(algebra, inputs, output, case), case, placement)

    facts = metadata["structural_facts"]
    assert "exterior_entries" in facts
    assert "exterior_work" in facts
    assert "minor_entries" not in facts
    assert "determinant_work" not in facts


def test_forced_exp_root_records_normally_selected_product_children():
    case = _case("bivector_exp")
    placement = ExecutionPlacement(selection=SelectionCase("forced_repository_private", "taylor"))
    algebra, inputs, output, _ = construct_case(case, placement)
    metadata = planning_metadata(plan_case(algebra, inputs, output, case), case, placement)

    assert metadata["selected_route"] == "taylor"
    assert metadata["child_selections"]
    assert {item["family"] for item in metadata["child_selections"]} == {"product"}
