"""Fixed-suite expansion, deterministic ordering, and checkpointing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .adapters import CLIFRA_ADAPTER, configure_clifra_runtime, route_assessments
from .cases import (
    AMORTIZATION_CASE_IDS,
    BenchmarkCase,
    ExecutionPlacement,
    SelectionCase,
    full_cases,
    smoke_cases,
)
from .engine import SCHEMA_VERSION, MeasurementConfig, measure_request
from .model import TIMING_MODES, BenchmarkRequest
from .provenance import capture_immutable_provenance, capture_runtime_provenance


@dataclass(frozen=True)
class CampaignPlan:
    semantic_cases: tuple[BenchmarkCase, ...]
    requests: tuple[BenchmarkRequest, ...]
    pruned: tuple[dict[str, object], ...]

    def summary(self) -> dict[str, object]:
        semantic_counts: dict[str, int] = {}
        placement_counts: dict[str, int] = {}
        counts: dict[str, int] = {}
        for case in self.semantic_cases:
            semantic_counts[case.family] = semantic_counts.get(case.family, 0) + 1
        for request in self.requests:
            selection = request.placement.selection
            route = selection.route or "normal"
            placement_key = "/".join((request.case.family, request.placement.device, request.placement.dtype))
            placement_counts[placement_key] = placement_counts.get(placement_key, 0) + 1
            key = "/".join((request.case.family, request.placement.device, request.placement.dtype, route))
            counts[key] = counts.get(key, 0) + 1
        reasons: dict[str, int] = {}
        for item in self.pruned:
            reason = str(item.get("category", item["reason"]))
            reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "semantic_case_count": len(self.semantic_cases),
            "semantic_cases_by_family": semantic_counts,
            "execution_row_count": len(self.requests),
            "execution_rows_by_family_device_dtype": placement_counts,
            "execution_rows_by_family_device_dtype_route": counts,
            "pruned_count": len(self.pruned),
            "pruned_by_reason": reasons,
        }


def _request_fingerprint(
    request,
    config,
    cpu_threads,
    interop_threads,
    source_fingerprint,
    campaign_seed,
):
    payload = {
        "schema_version": SCHEMA_VERSION,
        **request.to_dict(),
        "measurement": {"samples": config.samples, "warmups": config.warmups},
        "runtime": {"cpu_threads": cpu_threads, "interop_threads": interop_threads},
        "campaign_seed": campaign_seed,
        "measurement_source_sha256": source_fingerprint,
        "execution_strategy": "in_process",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def deterministic_order(requests: tuple[BenchmarkRequest, ...], seed: int) -> tuple[BenchmarkRequest, ...]:
    """Return an input-order-independent deterministic campaign shuffle."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("campaign seed must be a non-negative integer")
    ordered = sorted(requests, key=lambda request: request.row_id)
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def _checkpoint_rows(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    rows = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid checkpoint JSON on line {line_number}: {error}") from error
        fingerprint = row.get("request_fingerprint")
        if not isinstance(fingerprint, str):
            raise ValueError(f"checkpoint line {line_number} has no request_fingerprint")
        rows[fingerprint] = row
    return rows


def run_requests(
    requests: tuple[BenchmarkRequest, ...],
    config: MeasurementConfig,
    *,
    cpu_threads: int = 1,
    interop_threads: int = 1,
    progress: bool = False,
    campaign_seed: int = 1729,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    pruned: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    """Measure an ordered campaign directly in the current process."""

    configure_clifra_runtime(cpu_threads, interop_threads)
    ordered = deterministic_order(requests, campaign_seed)
    immutable_provenance = capture_immutable_provenance()
    source_fingerprint = str(immutable_provenance["source"]["measurement_source_sha256"])
    provenance_by_device = {
        device: capture_runtime_provenance(device, immutable_provenance)
        for device in sorted({request.placement.device for request in ordered})
    }
    completed_rows = _checkpoint_rows(checkpoint_path) if resume else {}
    rows = []
    skipped = 0
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_path.open("a", encoding="utf-8")
    try:
        for ordinal, request in enumerate(ordered, start=1):
            fingerprint = _request_fingerprint(
                request,
                config,
                cpu_threads,
                interop_threads,
                source_fingerprint,
                campaign_seed,
            )
            if fingerprint in completed_rows:
                row = completed_rows[fingerprint]
                skipped += 1
                state = "checkpoint"
            else:
                row = measure_request(
                    request,
                    CLIFRA_ADAPTER,
                    config,
                    request_fingerprint=fingerprint,
                    provenance=provenance_by_device[request.placement.device],
                )
                if checkpoint is not None:
                    checkpoint.write(json.dumps(row, sort_keys=True) + "\n")
                    checkpoint.flush()
                    os.fsync(checkpoint.fileno())
                state = row["status"]
            rows.append(row)
            if progress:
                print(f"[{ordinal}/{len(ordered)}] {row['row_id']}: {state}", file=sys.stderr, flush=True)
    finally:
        if checkpoint is not None:
            checkpoint.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "campaign": {
            "seed": campaign_seed,
            "request_count": len(ordered),
            "checkpoint_rows_reused": skipped,
            "ordering": "canonical row_id sort followed by random.Random(seed).shuffle",
            "execution_strategy": "in_process",
            "provenance_scope": "source/environment once per campaign; device/runtime once per device",
            "pruned": list(pruned),
        },
        "rows": rows,
    }


def run_cases(
    cases: tuple[BenchmarkCase, ...],
    modes: tuple[str, ...],
    config: MeasurementConfig,
    *,
    placement: ExecutionPlacement = ExecutionPlacement(),
    cpu_threads: int = 1,
    interop_threads: int = 1,
) -> dict[str, object]:
    requests = tuple(BenchmarkRequest(case, placement, mode) for case in cases for mode in modes)
    return run_requests(
        requests,
        config,
        cpu_threads=cpu_threads,
        interop_threads=interop_threads,
    )


def campaign_plan(
    devices: tuple[str, ...] = ("cpu",),
    *,
    dtypes: tuple[str, ...] | None = None,
    families: tuple[str, ...] = ("product", "bivector_exp", "action"),
    modes: tuple[str, ...] | None = None,
) -> CampaignPlan:
    """Expand semantic axes into normal and every feasible forced-root row."""

    cases = full_cases(families)
    requests = []
    pruned = []
    for case in cases:
        case_modes = modes or ("steady_forward", "forward_backward")
        if modes is None and case.case_id in AMORTIZATION_CASE_IDS:
            case_modes += ("planning_preparation", "first_invocation")
        for device in devices:
            device_dtypes = (
                dtypes if dtypes is not None else ("float32",) if device == "mps" else ("float32", "float64")
            )
            for dtype in device_dtypes:
                placement = ExecutionPlacement(dtype, device)
                assessments = route_assessments(case, placement)
                feasible = tuple(item["route"] for item in assessments if item["eligible"])
                if not feasible:
                    reasons = sorted({str(item["ineligible_reason"]) for item in assessments})
                    pruned.append(
                        {
                            "case_id": case.case_id,
                            "device": device,
                            "dtype": dtype,
                            "category": "capability_or_resource_limit",
                            "reason": "no_feasible_root_route: " + "; ".join(reasons),
                        }
                    )
                    continue
                placements = (placement,) + tuple(
                    ExecutionPlacement(dtype, device, SelectionCase("forced_repository_private", route))
                    for route in feasible
                )
                requests.extend(
                    BenchmarkRequest(case, selected_placement, mode)
                    for selected_placement in placements
                    for mode in case_modes
                )
    return CampaignPlan(cases, tuple(requests), tuple(pruned))


def smoke_requests(device: str, modes: tuple[str, ...], dtype: str = "float32"):
    requests = []
    for case in smoke_cases():
        placement = ExecutionPlacement(dtype, device)
        if device in {"mps", "cuda"}:
            placements = (placement,)
        else:
            feasible = tuple(item["route"] for item in route_assessments(case, placement) if item["eligible"])
            placements = (placement,) + tuple(
                ExecutionPlacement(dtype, device, SelectionCase("forced_repository_private", route))
                for route in feasible
            )
        requests.extend(BenchmarkRequest(case, item, mode) for item in placements for mode in modes)
    return tuple(requests)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run clifra's fixed routed-family benchmark suite")
    parser.add_argument("--output", type=Path, required=True, help="authoritative JSON artifact path")
    parser.add_argument("--suite", choices=("full", "smoke"), default="full")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--mode", action="append", choices=sorted(TIMING_MODES), dest="modes")
    parser.add_argument("--case-id", action="append", help="run only these exact semantic case ids")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--campaign-seed", type=int, default=1729)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output artifact")
    parser.add_argument("--describe", action="store_true", help="print matrix counts and pruning without execution")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    devices = (args.device,)
    modes = tuple(args.modes or ("steady_forward", "forward_backward"))
    pruned = ()
    if args.suite == "smoke":
        requests = tuple(request for device in devices for request in smoke_requests(device, modes, args.dtype))
        summary = {"semantic_case_count": len(smoke_cases()), "execution_row_count": len(requests)}
    else:
        plan = campaign_plan(
            devices,
            dtypes=(args.dtype,),
            modes=None if args.modes is None else modes,
        )
        requests, pruned, summary = plan.requests, plan.pruned, plan.summary()
    if args.case_id:
        requested_ids = set(args.case_id)
        requests = tuple(item for item in requests if item.case.case_id in requested_ids)
        missing = requested_ids - {item.case.case_id for item in requests}
        if missing:
            raise ValueError(f"unknown or fully pruned case ids: {sorted(missing)!r}")
    if args.describe:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to replace immutable artifact {args.output}; pass --overwrite explicitly")
    artifact = run_requests(
        requests,
        MeasurementConfig(args.samples, args.warmups),
        cpu_threads=args.cpu_threads,
        interop_threads=args.interop_threads,
        progress=True,
        campaign_seed=args.campaign_seed,
        checkpoint_path=args.checkpoint,
        resume=not args.no_resume,
        pruned=pruned,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(artifact['rows'])} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
