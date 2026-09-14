# Relativistic Beamline Inverse Design

This experiment optimizes a spatially varying electromagnetic bivector field in
\(Cl(1,3)\) to steer a relativistic particle bunch through three offset
apertures and onto a compact target. Natural units \(c=q/m=1\) are used.
Four-velocity updates are Lorentz rotor actions; worldlines are integrated in
proper time.

![Initial and optimized relativistic beamline](../assets/research/relativistic_beamline_design/result.png)

## Experiment

Eleven longitudinal control sites parameterize six field channels with local
\(C^1\) cubic interpolation along the beamline. The optimization uses 25
particles. Its objective combines aperture and target geometry, focusing,
direction and energy terms, forward-motion constraints, and compact field
regularization. There are no separate centroid-equality losses for the
intermediate apertures or target. Softened aperture and target-region penalties
favor interior passage, while focusing, direction, and energy terms shape the
downstream bunch. The region centers define those penalties, but gate and target
centroid errors are diagnostics rather than acceptance limits.
The softened penalties use normalized-radius limits of 0.92 at the gates and
0.88 at the target; acceptance checks require passage inside the actual
unit-radius regions.

The restored field is then evaluated without retraining on an off-grid
\(9\times9\) transverse phase-space grid containing 81 held-out particles.
A separate \(h\), \(h/2\), \(h/4\) refinement study evaluates sensitivity to
proper-time discretization.

| Quantity | Optimized / held-out result |
| --- | ---: |
| Optimized gate max normalized radii | 0.687, 0.674, 0.682 |
| Held-out gate max normalized radii | 0.743, 0.687, 0.878 |
| Held-out target pass fraction | 100% |
| Held-out target max normalized radius | 0.829 |
| Held-out target RMS spread | 28.3 mm |
| Held-out target centroid error (diagnostic) | 25.2 mm |
| Held-out mean \(\gamma\) | 1.650 |
| Held-out mass-shell max error | \(4.8\times10^{-14}\) |

Successive crossing differences decrease under refinement. From \(h/2\) to
\(h/4\), the held-out RMS change is 3.88 mm at the gates and 3.34 mm at the
target. The refinement is reported as observed convergence behavior rather than
as a formal convergence-order estimate.

![Relativistic beamline diagnostics](../assets/research/relativistic_beamline_design/diagnostics.png)

Recorded default run summary:

```text
setup         25 optimization particles | 11 field sites | 144 steps at h=0.042
[   0/520] loss=2.8683e+03 aperture=1.9907 focus=0.1556 gamma=1.539
[ 520/520] loss=1.3227e-01 aperture=0.0099 focus=0.0201 gamma=1.650
validation   81 held-out particles | target 100.0% | spread 28.31 mm | centroid 25.23 mm
refinement   target ΔRMS 6.26→3.34 mm | gate ΔRMS 7.90→3.88 mm
numerical    gamma 1.6498 | mass shell 4.8e-14
optimization best 1.323e-01 at 520 | final 1.323e-01
```

## Run

```bash
uv run --group viz research/transformation_fields/examples/relativistic_beamline_design.py
```

Add `--live` for the interactive particle-bundle view. Saved artifacts are
written to `outputs/relativistic_beamline_design/`.
