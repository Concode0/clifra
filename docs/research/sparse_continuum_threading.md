# Sparse-Constraint Continuum Threading

This experiment fits a continuous material-space transformation field from
three gate poses and one tip pose. No target centerline or material-to-gate
correspondence is supplied to the optimizer. A \(Cl(4,1)\) conformal model
supplies an SE(3) generator subspace, while an RBF sampler varies those
generators along the persistent material coordinate.

![Dense continuum configuration](../assets/research/sparse_continuum_threading/result.png)

## Experiment

The optimization uses a uniform material grid with 42 cross-sections and four
surface samples each, 210 points in total. It does not snap samples to gates.
The undeformed rod length is a fixed 1.483 m scene parameter.
At each step, the current predicted centerline determines the first forward
segment crossing each gate plane. Segment selection is detached, while the
crossing point is interpolated differentiably within that segment. Ten ordered
local actions deform the rod while losses enforce gate-center, gate-alignment,
and tip constraints together with collision, strain, curvature, base, and
smoothness terms.

After optimization, the same field parameters are evaluated directly on
180 cross-sections with twelve surface samples each: 2,340 points with no
additional optimizer step.

The dense evaluation reports:

| Quantity | Result |
| --- | ---: |
| Gate crossing center offset | 1.80 mm |
| Tip position error | 0.88 mm |
| Inferred gate material coordinates | 0.2195, 0.5322, 0.8036 |
| Coarse → dense centerline discrepancy | 1.46 mm |
| Minimum obstacle clearance | 65.1 mm |
| Maximum axial strain | 0.056 |
| Indexed inverse error | \(4.7\times10^{-15}\) |
| Cross-section rigidity error | \(7.6\times10^{-14}\) |

All acceptance checks pass for sparse threading, geometric validity, analytic
inverse and rigidity, and zero-shot dense transfer. Threading requires genuine
forward gate-plane crossings on both grids. Clearance and curvature are checked
at sampled centerline sections, so they are not continuous-segment guarantees.

![Continuum diagnostics](../assets/research/sparse_continuum_threading/diagnostics.png)

Recorded default run summary:

```text
setup         3 gates + 1 tip pose | 14 RBF controls | 210 optimization points | seed 17
[   0/800] loss=2.488e+02 gate=0.5237 tip=0.6042 orient=159.5° clear=+0.0336 strain=0.010
[ 800/800] loss=1.215e-02 gate=0.0023 tip=0.0010 orient=6.3° clear=+0.0651 strain=0.054
primary      gate 0.0025 m | tip 0.0009 m | clearance +0.0654 m
validation   210→2340 points | gate 0.0018 m | transfer Δ 0.0015 m
gate s        0.2195, 0.5322, 0.8036
numerical    inverse 4.7e-15 | rigidity 7.6e-14 | strain 0.056
optimization best 1.064e-02 at 799 | final 1.215e-02
```

## Run

```bash
uv run --group viz research/transformation_fields/examples/sparse_continuum_threading.py
```

Add `--live` for the interactive optimization view. The canonical saved
artifacts are written to `outputs/sparse_continuum_threading/`.
