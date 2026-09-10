# Use experimental analysis

`clifra.analysis.experimental` contains exploratory measurements whose
interpretation depends on representation choices. Keep these separate from
diagnostics conditioned on a declared algebra.

`NeighborhoodBivectorAnalyzer` selects nearest neighbors using Euclidean distance
over all canonical coefficients. It then wedges the grade-1 parts of each
point-neighbor pair and normalizes by positive coefficient length. These are
wedges of positions, not wedges of displacement vectors; changing the origin
can change the result.

```python
import torch
from clifra import make_algebra
from clifra.analysis.experimental import (
    NeighborhoodBivectorAnalyzer,
    compare_coordinate_lifts,
)

coordinates = torch.tensor(
    [[1., 0.], [0.8, 0.6], [0., 1.], [-0.6, 0.8]], dtype=torch.float64
)
algebra = make_algebra(2, dtype=torch.float64)
observations = algebra.layout((1,)).full(coordinates)
analyzer = NeighborhoodBivectorAnalyzer(algebra, k=2)
means = analyzer.mean_neighbor_bivectors(observations)
assert means.shape == observations.shape
assert torch.isfinite(means).all()
alignment = analyzer.neighbor_bivector_alignment(observations)
assert 0. <= alignment <= 1. + 1e-12

lifts = compare_coordinate_lifts(coordinates, p=1, q=1, k=2)
assert lifts["original"]["algebra_signature"] == (1, 1)
assert lifts["positive_count_extension"]["algebra_signature"] == (2, 1)
assert lifts["negative_count_extension"]["algebra_signature"] == (1, 2)
```

Alignment compares absolute coefficient dot products of normalized wedges.
Oppositely oriented wedges can cancel in `mean_neighbor_bivectors` even when
their alignment is high. Zero wedges remain zero. Neighbor selection is discrete
and uses a dense pairwise distance matrix, so both differentiability and memory
cost need consideration before using a score in a training loop.

`compare_coordinate_lifts` accepts raw `[N, p+q]` coordinates. The positive-count
extension inserts a constant after the positive coordinates; the negative-count
extension appends it after the original negative coordinates. Both preserve the
original positive and negative directions. The report contains signatures and
scores, without an automatic preferred representation or threshold verdict.

## Probe candidate signatures

`SignatureProbeAnalyzer` trains small probes on detached raw `[N, D]`
observations. It may reduce features by PCA, then reports candidate active
positive/negative counts based on trained bivector parameter energies. Those
counts are evidence about this probe procedure, not a recovery theorem for the
data's intrinsic metric. Training work is controlled by `num_probes`,
`probe_epochs`, `probe_lr`, and `max_probe_features`; `k` controls neighborhoods.

The following optional smoke example deliberately uses little training. Its
output should not be treated as a converged representation decision.

```python title="Optional signature probe"
from clifra.analysis.experimental import SignatureProbeAnalyzer

torch.manual_seed(7)
probe = SignatureProbeAnalyzer(
    dtype=torch.float64, max_probe_features=2,
    num_probes=1, probe_epochs=1, k=2,
)
candidate = probe.analyze(coordinates)
assert candidate.pca_output_width is None
```

`analyze_bootstrap` reports counts across resampled runs and returns the first
replicate with the modal candidate tuple, rather than refitting on the original
data. Its seed controls resampling indices; set PyTorch's RNG separately for
probe training. PCA, initialization, the energy threshold, and the training
budget all affect interpretation. These probes detach the caller's graph and
are not a differentiable signature-selection operation.
