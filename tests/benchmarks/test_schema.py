import json
from pathlib import Path

from benchmarks.adapters import CLIFRA_ADAPTER
from benchmarks.cases import BenchmarkCase, ExecutionPlacement, LayoutCase
from benchmarks.engine import SCHEMA_VERSION, MeasurementConfig, measure_request
from benchmarks.model import BenchmarkRequest
from benchmarks.run import run_cases


def test_schema_document_and_emitted_row_share_version_and_required_fields():
    schema = json.loads((Path(__file__).parents[2] / "benchmarks" / "result.schema.json").read_text())
    case = BenchmarkCase(
        case_id="product.schema.tiny",
        operation="wedge",
        signature=(2, 0, 0),
        inputs=(LayoutCase((1,), leading_shape=(1,)),) * 2,
        output=LayoutCase((2,)),
    )
    row = measure_request(
        BenchmarkRequest(case, ExecutionPlacement(), "steady_forward"),
        CLIFRA_ADAPTER,
        MeasurementConfig(samples=1, warmups=0),
    )

    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert schema["$defs"]["row"]["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert set(schema["$defs"]["row"]["required"]) == set(row)


def test_in_process_artifact_validates_against_versioned_schema():
    schema = json.loads((Path(__file__).parents[2] / "benchmarks" / "result.schema.json").read_text())
    case = BenchmarkCase(
        case_id="product.schema.artifact",
        operation="geometric_product",
        signature=(2, 0, 0),
        inputs=(LayoutCase((1,), leading_shape=(1,)),) * 2,
        output=LayoutCase((0, 2)),
    )
    artifact = run_cases((case,), ("steady_forward",), MeasurementConfig(samples=1, warmups=0))

    assert set(artifact) == set(schema["required"])
    assert artifact["schema_version"] == schema["properties"]["schema_version"]["const"]
    assert set(artifact["campaign"]) == set(schema["properties"]["campaign"]["required"])
    assert all(set(row) == set(schema["$defs"]["row"]["required"]) for row in artifact["rows"])
    assert all(row["status"] == "ok" for row in artifact["rows"])
    assert all("correctness" not in row for row in artifact["rows"])
