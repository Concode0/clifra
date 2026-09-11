"""Internal benchmark CLI. Run with uv run --group dev python -m benchmarks.run."""

import argparse
import fnmatch
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch

from clifra.core import AlgebraContext
from clifra.core._kernel.planning.policy import NoAvailableRouteError

from .adapter import Unsupported, make_inputs, prepare, temporary_workarounds
from .cases import SMOKE, TRACKS, core_cases, sweep
from .correctness import check, gate
from .timing import measure, timed, workload

ROUTES = {
    "product": ("full_table", "sparse"),
    "bivector_exp": ("closed", "taylor", "left_matrix_exp"),
    "action": ("vector_matrix", "rotor_product", "full_action_matrix"),
}


def availability(device, dtype):
    d = torch.device(device)
    if d.type == "cuda" and (not torch.cuda.is_available() or (d.index or 0) >= torch.cuda.device_count()):
        return "cuda_device_unavailable"
    if d.type == "mps" and not torch.backends.mps.is_available():
        return "mps_device_unavailable"
    if d.type == "mps" and dtype == "float64":
        return "mps_does_not_support_float64"
    if d.type not in {"cpu", "cuda", "mps"}:
        return "unsupported_device_type"
    return None


def environment(options):
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        clifra_version = importlib.metadata.version("clifra")
    except importlib.metadata.PackageNotFoundError:
        clifra_version = "source-tree"
    try:
        cpu_model = (
            subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            if platform.system() == "Darwin"
            else platform.processor()
        )
    except (OSError, subprocess.CalledProcessError):
        cpu_model = platform.processor()
    return {
        "record": "environment",
        "schema_version": 2,
        "created_at_unix": time.time(),
        "revision": git("rev-parse", "HEAD"),
        "harness_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
        "dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "clifra_version": clifra_version,
        "cpu_model": cpu_model,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_version": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "mps_available": torch.backends.mps.is_available(),
        "options": vars(options),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH")
        },
    }


def execute(case, route, options):
    row = {
        "record": "result",
        "schema_version": 2,
        "semantic_case_id": case.id,
        "semantic": asdict(case),
        "implementation_id": f"clifra/{case.family}/{route}/{options.storage}",
        "requested_route": route,
        "policy_selected_route": None,
        "device": options.device,
        "dtype": options.dtype,
        "mode": options.mode,
        "storage": options.storage,
        "status": "error",
        "correctness": {"status": "not_run"},
        "timings": {},
        "compile": {"status": "not_requested"},
    }
    stage = "availability"
    try:
        reason = availability(options.device, options.dtype)
        if reason:
            raise Unsupported(reason)
        dtype = getattr(torch, options.dtype)
        torch.set_num_threads(options.threads)
        torch.set_num_interop_threads(1)
        stage = "setup"
        algebra, row["timings"]["context_ms"] = timed(
            lambda: AlgebraContext(*case.signature, device=options.device, dtype=dtype), options.device
        )
        row["resource_limits"] = asdict(algebra.resource_limits)
        # Preflight the independent reference before allocating large executor buffers.
        from .correctness import reference

        reference(algebra, case, options.storage, options.reference_terms, options.reference_order)
        (args, seed), row["timings"]["inputs_ms"] = timed(
            lambda: make_inputs(algebra, case, options.storage, options.max_bytes), options.device
        )
        row.update(seed=seed, shapes=[list(x.shape) for x in args], strides=[list(x.stride()) for x in args])
        stage = "planning"
        operation, metadata, stages = prepare(algebra, case, route, options.storage)
        row["implementation"] = metadata
        row["temporary_workarounds"] = temporary_workarounds(case, metadata, args)
        row["policy_comparison_eligible"] = not row["temporary_workarounds"]
        row["timings"].update(stages)
        row["policy_selected_route"] = metadata["policy_selected_route"]
        backward = options.mode in {"backward", "compile-backward"}
        row["gradient_mode"] = "forward_backward" if backward else "no_grad_forward"
        row["differentiable_inputs"] = list(range(len(args))) if backward else []
        stage = "correctness"
        row["correctness"], oracle = gate(
            algebra,
            case,
            options.storage,
            operation,
            args,
            backward,
            max_terms=options.reference_terms,
            max_order=options.reference_order,
        )
        target = operation
        if options.mode.startswith("compile"):
            stage = "compile"
            row["compile"] = {
                "status": "started",
                "backend": options.backend,
                "fullgraph": True,
                "cache": "fresh_worker_and_inductor_directory",
            }
            target, row["compile"]["wrap_ms"] = timed(
                lambda: torch.compile(operation, backend=options.backend, fullgraph=True), options.device
            )
            leaves = tuple(x.detach().requires_grad_(backward) for x in args)
            with torch.set_grad_enabled(backward):
                output, row["compile"]["first_forward_ms"] = timed(lambda: target(*leaves), options.device)
                if backward:
                    weights = torch.linspace(
                        0.5, 1.5, output.numel(), device=output.device, dtype=output.dtype
                    ).reshape(output.shape)
                    loss = (output * weights).mean()
                    _, row["compile"]["first_backward_ms"] = timed(loss.backward, options.device)
            stage = "compiled_correctness"
            row["correctness"] = check(target, oracle, args, backward, dtype)
            row["compile"]["status"] = "passed"
        stage = "execution"
        step = workload(target, args, backward, probe=operation)
        for _ in range(options.warmup):
            step()
        from torch._dynamo.utils import counters

        graphs = counters["stats"]["unique_graphs"]
        row["timings"]["steady"] = measure(step, options.device, 0, options.samples, options.repetitions)
        row["timings"]["steady"]["warmup_iterations"] = options.warmup
        if options.mode.startswith("compile") and counters["stats"]["unique_graphs"] != graphs:
            del row["timings"]["steady"]
            raise RuntimeError("recompilation_during_steady_measurement")
        row["status"] = "ok"
    except (Unsupported, NoAvailableRouteError) as exc:
        row.update(status="unsupported", reason=str(exc), stage=stage)
    except AssertionError as exc:
        row.update(status="correctness_failed" if "correctness" in stage else "error", reason=str(exc), stage=stage)
        if "correctness" in stage:
            row["correctness"] = {"status": "failed", "reason": str(exc)}
        if stage == "compiled_correctness":
            row["compile"]["status"] = "correctness_failed"
    except Exception as exc:
        message = str(exc)
        unavailable_kernel = isinstance(exc, (RuntimeError, NotImplementedError)) and any(
            text in message.lower()
            for text in (
                "not implemented for 'half'",
                "not implemented for 'bfloat16'",
                "does not support bfloat16",
                "does not support float64",
                "not implemented for the mps device",
            )
        )
        row.update(
            status="unsupported" if unavailable_kernel else "error", reason=f"{type(exc).__name__}: {exc}", stage=stage
        )
    return row


def worker(connection, case, route, options):
    # Per-job isolation bounds runaway qualification and prevents compiler-cache contamination.
    with tempfile.TemporaryDirectory(prefix="clifra-benchmark-") as directory:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = directory
        try:
            connection.send(execute(case, route, options))
        finally:
            connection.close()


def isolated(case, route, options):
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(writer, case, route, options))
    process.start()
    writer.close()
    try:
        if reader.poll(options.timeout):
            try:
                result = reader.recv()
            except EOFError:
                result = {"status": "error", "reason": "worker_exited_without_result"}
        else:
            result = {"status": "error", "reason": "worker_timeout"}
    finally:
        if process.is_alive():
            process.join(1)
        if process.is_alive():
            process.terminate()
        process.join()
        reader.close()
    return {
        "record": "result",
        "semantic_case_id": case.id,
        "requested_route": route,
        "device": options.device,
        "dtype": options.dtype,
        "mode": options.mode,
        **result,
    }


def observations(rows):
    notes = []
    groups = {}
    for row in rows:
        if (
            row["status"] == "ok"
            and row.get("policy_comparison_eligible", True)
            and not row.get("temporary_workarounds")
        ):
            groups.setdefault(row["semantic_case_id"], []).append(row)
    for id, group in groups.items():
        default = next((r for r in group if r["requested_route"] == "default"), None)
        if default is None:
            continue
        best = min(group, key=lambda r: r["timings"]["steady"]["median_ms"])
        d, b = default["timings"]["steady"], best["timings"]["steady"]
        if (
            best["implementation"]["route"] != default["policy_selected_route"]
            and b["p90_ms"] < d["p10_ms"]
            and b["median_ms"] * 1.2 < d["median_ms"]
        ):
            notes.append(
                {
                    "case": id,
                    "selected": default["policy_selected_route"],
                    "alternative": best["implementation"]["route"],
                    "median_ratio": d["median_ms"] / b["median_ms"],
                    "interpretation": "candidate crossover; confirm across runs and devices",
                }
            )
    return notes


def compare(paths):
    def read(path):
        with open(path) as stream:
            records = [json.loads(line) for line in stream]
        return records[0], [r for r in records if r.get("record") == "result" and r["status"] == "ok"]

    old_env, old = read(paths[0])
    new_env, new = read(paths[1])
    keys = (
        "cpu_model",
        "platform",
        "machine",
        "processor",
        "torch",
        "cuda_version",
        "cuda_devices",
        "environment",
        "float32_matmul_precision",
        "tf32",
        "deterministic_algorithms",
    )
    compatible = all(old_env.get(k) == new_env.get(k) for k in keys)
    compatible &= all(
        old_env["options"].get(k) == new_env["options"].get(k)
        for k in ("threads", "backend", "repetitions", "warmup", "samples")
    )

    def key(r):
        return json.dumps(
            [r["semantic"], r["implementation_id"], r["device"], r["dtype"], r["mode"], r["shapes"], r["strides"]],
            sort_keys=True,
        )

    baseline = {key(r): r for r in old}
    pairs = []
    for row in new:
        previous = baseline.get(key(row))
        if previous:
            pairs.append(
                {
                    "case": row["semantic_case_id"],
                    "implementation": row["implementation_id"],
                    "median_ratio": row["timings"]["steady"]["median_ms"] / previous["timings"]["steady"]["median_ms"],
                    "route_changed": row["implementation"]["route"] != previous["implementation"]["route"],
                    "temporary_workarounds_changed": row.get("temporary_workarounds", [])
                    != previous.get("temporary_workarounds", []),
                    "policy_comparison_eligible": row.get("policy_comparison_eligible", True)
                    and previous.get("policy_comparison_eligible", True),
                }
            )
    print(json.dumps({"environment_compatible": compatible, "pairs": pairs}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("smoke", "core", "qualification"), default="core")
    parser.add_argument("--family")
    parser.add_argument("--case", action="append", default=[], help="repeatable case ID/glob filter")
    parser.add_argument("--sweep", choices=TRACKS)
    parser.add_argument("--dimensions", default="2,3,4", help="comma-separated dimensions for a sweep")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64", "float16", "bfloat16"), default="float32")
    parser.add_argument("--mode", choices=("eager", "backward", "compile", "compile-backward"), default="eager")
    parser.add_argument("--backend", choices=("inductor", "aot_eager"), default="inductor")
    parser.add_argument("--routes", default="default", help="default, all, or a comma-separated route list")
    parser.add_argument("--storage", choices=("compact", "canonical"), default="compact")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--reference-terms", type=int, default=200_000)
    parser.add_argument("--reference-order", type=int, default=128, help="maximum independent exponential matrix order")
    parser.add_argument("--output", default="benchmarks/results/latest.jsonl")
    parser.add_argument("--list", action="store_true", help="expand manifest without running")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"))
    options = parser.parse_args(argv)
    if options.compare:
        return compare(options.compare)
    if (
        min(
            options.samples,
            options.repetitions,
            options.threads,
            options.max_bytes,
            options.reference_terms,
            options.reference_order,
        )
        < 1
        or options.warmup < 0
        or options.timeout <= 0
    ):
        parser.error("sample, repetition, thread and budget counts must be positive; warmup must be nonnegative")
    cases = (
        list(sweep(options.sweep, tuple(map(int, options.dimensions.split(",")))))
        if options.sweep
        else list(core_cases())
    )
    if options.tier == "smoke" and not options.sweep:
        cases = [c for c in cases if c.id in SMOKE]
    cases = [
        c
        for c in cases
        if (not options.family or c.family == options.family)
        and (not options.case or any(fnmatch.fnmatchcase(c.id, pattern) for pattern in options.case))
    ]
    if not cases:
        parser.error("no cases matched")
    if options.list:
        for case in cases:
            print(json.dumps(asdict(case)))
        print(f"{len(cases)} cases", file=__import__("sys").stderr)
        return 0
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Refuse accidental replacement of a release baseline.
    with output.open("x") as stream:
        stream.write(json.dumps(environment(options)) + "\n")
        for case in cases:
            routes = (
                ("default", *ROUTES.get(case.family, ()))
                if options.routes == "all"
                else tuple(options.routes.split(","))
            )
            for route in routes:
                row = isolated(case, route, options)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                timing = row.get("timings", {}).get("steady")
                latency = (
                    f"{timing['median_ms']:.3g} ms [{timing['p10_ms']:.3g}, {timing['p90_ms']:.3g}]"
                    if timing
                    else row.get("reason", "")[:150]
                )
                selected = row.get("implementation", {}).get("route")
                label = f"{route}->{selected}" if route == "default" and selected else route
                if row.get("temporary_workarounds"):
                    latency += " [temporary workaround; policy excluded]"
                print(f"{row['status']:18} {case.id} {label:26} {latency}", flush=True)
        notes = observations(rows)
        stream.write(json.dumps({"record": "policy_observations", "observations": notes}) + "\n")
    print(
        f"Results: {output}; "
        + ", ".join(
            f"{status}={sum(r['status'] == status for r in rows)}"
            for status in ("ok", "unsupported", "correctness_failed", "error")
        )
    )
    for note in notes:
        print("Policy observation:", json.dumps(note))
    return int(
        any(
            r["status"] in {"error", "correctness_failed"}
            or options.tier == "qualification"
            and r["requested_route"] == "default"
            and r["status"] != "ok"
            for r in rows
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
