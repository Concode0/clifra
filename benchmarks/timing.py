"""Synchronized wall-clock samples; no calibration or retained autograd graphs."""

import statistics
import time

import torch


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(fn, device):
    synchronize(device)
    start = time.perf_counter_ns()
    value = fn()
    synchronize(device)
    return value, (time.perf_counter_ns() - start) / 1e6


def stats(samples, repetitions=1):
    ordered = sorted(samples)

    def percentile(q):
        pos = (len(ordered) - 1) * q
        i = int(pos)
        return ordered[i] + (ordered[min(i + 1, len(ordered) - 1)] - ordered[i]) * (pos - i)

    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "p10_ms": percentile(0.1),
        "p90_ms": percentile(0.9),
        "count": len(samples),
        "repetitions": repetitions,
    }


def workload(fn, args, backward, *, probe=None):
    leaves = tuple(x.detach().requires_grad_(backward) for x in args)
    with torch.no_grad():
        shape_probe = (fn if probe is None else probe)(*leaves)
    weights = torch.linspace(0.5, 1.5, shape_probe.numel(), device=shape_probe.device, dtype=shape_probe.dtype).reshape(
        shape_probe.shape
    )
    del shape_probe

    def step():
        if backward:
            for x in leaves:
                x.grad = None
            (fn(*leaves) * weights).mean().backward()
        else:
            with torch.no_grad():
                fn(*leaves)

    return step


def measure(step, device, warmup, samples, repetitions):
    for _ in range(warmup):
        step()

    def block():
        for _ in range(repetitions):
            step()

    values = [timed(block, device)[1] / repetitions for _ in range(samples)]
    return {**stats(values, repetitions), "warmup_iterations": warmup, "metric": "synchronized_wall_ms_per_iteration"}
