# Analyze spectra, commutators, and transformations

Stable analyzers describe observations in an explicitly supplied algebra.
They accept nonempty canonical `[N, algebra.dim]` floating-point tensors and
convert them to the algebra's dtype and device. Expand compact observations
explicitly. For tensors with channels or other leading axes, decide whether
to average them or treat them as separate observations before analysis.

```python
import torch
from clifra import make_algebra
from clifra.analysis import (
    CommutatorAnalyzer,
    SpectralAnalyzer,
    TransformationDiagnosticsAnalyzer,
)

algebra = make_algebra(3, dtype=torch.float64)
coordinates = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
observations = algebra.layout((1,)).full(coordinates)
spectral = SpectralAnalyzer(algebra).analyze(observations)
torch.testing.assert_close(
    spectral.grade_coefficient_energy, coordinates.new_tensor([0., 1., 0., 0.])
)
assert spectral.left_multiplication_eigenvalue_magnitudes is not None

commutators = CommutatorAnalyzer(algebra).analyze(observations)
transformations = TransformationDiagnosticsAnalyzer(algebra).analyze(observations)

assert abs(commutators.mean_commutator_norm - 1.) < 1e-12
assert transformations.odd_grade_energy_fraction == 1.
```

Grade coefficient energy averages the squared coefficients of each observation,
not the squared coefficients of the sample mean. The left-multiplication
spectrum instead describes the operator $x\mapsto\bar a x$ for the complete
sample mean $\bar a$. Its returned values are eigenvalue magnitudes in descending
order; they are neither singular values nor a spectrum of the covariance matrix.
`mean_bivector` is the canonical grade-2 projection of that same mean.

The commutator diagnostics use the full bracket $[a,b]=ab-ba$, following the core
`commutator_product` convention without a factor of one half. Mean commutator norm measures
the coefficient norm of $[a_i,\bar a]$, averaged over observations. The vector
pair measurement and basis-bivector ratios answer different questions; they
measure asymmetry and basis alignment within the declared signature rather than classifying an algebra or inferring a signature from the observations.

To test closure of a specific bivector span, supply its canonical blade indices:

```python
closure = CommutatorAnalyzer(algebra).bivector_bracket_closure([3, 5, 6])
assert closure["mean_relative_closure_residual"] == 0.
structure = closure["projected_bracket_coefficients"]
torch.testing.assert_close(structure, -structure.transpose(0, 1))
```

Indices refer to canonical bitmasks, not compact lane positions. The requested
span is preserved in the supplied order. This method accepts at most 15 basis
bivectors and reports projected bracket coefficients and closure residuals.

## Handle unavailable measurements

Expensive full multiplication matrices and products have analysis-specific
resource guards. A skipped measurement is `None`, with a reason in `skipped`;
it is not a measured zero. Transformation diagnostics also report `None` for
reflection scores along null basis directions, where the inverse is undefined.

```python
degenerate = make_algebra(1, 0, 1, dtype=torch.float64)
data = degenerate.layout((1,)).full(coordinates[:, :2])

diagnostics = TransformationDiagnosticsAnalyzer(degenerate).analyze(data)
null_score = next(
    item["score"] for item in diagnostics.basis_reflection_marginal_scores
    if item["direction"] == 1
)

assert null_score is None
assert "basis_reflection_marginal_scores" in diagnostics.skipped
```

Reflection marginal scores are normalized coefficient-space distances under
basis reflections. Vector energies and odd-grade fractions are also coefficient
measurements, not signed quadratic forms or threshold-based symmetry verdicts.
Tensor-valued measurements can retain gradients, but result objects also contain
Python scalars, discrete choices, and optional measurements. For a training loss,
call the specific differentiable measurement you intend to optimize rather than
treating an entire diagnostic report as a loss.
