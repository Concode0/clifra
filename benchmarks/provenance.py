"""Environment provenance captured alongside every raw timing row."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _command(*args: str) -> str | None:
    try:
        return (
            subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True, timeout=3).stdout.strip() or None
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git() -> dict[str, object]:
    status = _command("git", "status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": _command("git", "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _source() -> dict[str, object]:
    """Fingerprint production changes and repository-private harness sources."""

    commit = _command("git", "rev-parse", "HEAD") or "unknown"
    production_diff = _command("git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", "clifra") or ""
    production_hash = hashlib.sha256(production_diff.encode()).hexdigest()
    benchmark_hash = hashlib.sha256()
    for path in sorted((ROOT / "benchmarks").iterdir()):
        if path.is_file() and path.suffix in {".py", ".json"}:
            benchmark_hash.update(str(path.relative_to(ROOT)).encode())
            benchmark_hash.update(b"\0")
            benchmark_hash.update(path.read_bytes())
            benchmark_hash.update(b"\0")
    harness_digest = benchmark_hash.hexdigest()
    combined = hashlib.sha256(f"{commit}\0{production_hash}\0{harness_digest}".encode()).hexdigest()
    return {
        "base_git_commit": commit,
        "production_diff_sha256": production_hash,
        "benchmark_sources_sha256": harness_digest,
        "measurement_source_sha256": combined,
    }


def _device(device: str) -> dict[str, object]:
    cpu_brand = _command("sysctl", "-n", "machdep.cpu.brand_string")
    hardware_model = _command("sysctl", "-n", "hw.model")
    if platform.system() == "Darwin" and (cpu_brand is None or hardware_model is None):
        hardware = _command("system_profiler", "SPHardwareDataType") or ""
        fields = {}
        for line in hardware.splitlines():
            key, separator, value = line.strip().partition(":")
            if separator and key in {"Model Name", "Model Identifier", "Chip"}:
                fields[key] = value.strip()
        cpu_brand = cpu_brand or fields.get("Chip")
        hardware_model = hardware_model or fields.get("Model Identifier") or fields.get("Model Name")
    if cpu_brand is None and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_brand = line.partition(":")[2].strip()
                break
    result = {
        "type": device,
        "identity": cpu_brand or platform.processor() or platform.machine(),
        "hardware_model": hardware_model,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "identity": torch.cuda.get_device_name(index),
                "cuda_device_index": index,
                "cuda_capability": list(torch.cuda.get_device_capability(index)),
                "cuda_total_memory_bytes": properties.total_memory,
                "cuda_multiprocessor_count": properties.multi_processor_count,
            }
        )
    return result


def capture_immutable_provenance() -> dict[str, object]:
    """Capture campaign-wide source and environment facts once."""

    try:
        clifra_version = importlib.metadata.version("clifra")
    except importlib.metadata.PackageNotFoundError:
        clifra_version = None
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "git": _git(),
        "source": _source(),
        "versions": {
            "clifra": clifra_version,
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "torch_cuda_runtime": torch.version.cuda,
            "torch_cudnn": torch.backends.cudnn.version(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "executable": sys.executable,
    }


def capture_runtime_provenance(device: str, immutable: dict[str, object]) -> dict[str, object]:
    """Combine campaign-wide facts with device and current-runtime facts."""

    result = deepcopy(immutable)
    result.update(
        {
            "process_id": os.getpid(),
            "device": _device(device),
            "threads": {
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            },
        }
    )
    return result


def capture_provenance(device: str) -> dict[str, object]:
    """Capture complete provenance for a standalone measurement."""

    return capture_runtime_provenance(device, capture_immutable_provenance())


def main() -> int:
    print(json.dumps(capture_provenance("cuda"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
