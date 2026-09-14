# Structural Work Models and Policy Calibration

clifra plans an operation before it sees the coefficient values or leading
tensor dimensions used at execution time. When several built-in executors can
implement the same request, `DefaultPolicy` therefore cannot be a latency
predictor over runtime inputs. It needs a static comparison derived from the
declared algebraic structure.

The policy is built in two stages:

1. derive an unweighted **structural work profile** from the algorithms
   themselves;
2. calibrate a small shared score over those structural coordinates using
   forced-route benchmark measurements.

The distinction matters. Benchmark timings choose coefficients and crossover
boundaries; they do not define the structural features. Conversely, a
structural count is not claimed to be an elapsed-time estimate.

This page describes that methodology. The scores are private planning
heuristics, not public performance guarantees or stable inspection APIs. See
[Planned Execution and Resources](planning-policy-injection.md) for the public
planning boundary and [Bivector Exponentials and Actions](bivector-exponential.md)
for the numerical algorithms summarized here.

## The planning boundary

Route selection uses information fixed by the request: the signature,
operation, layouts, dtype, and structural counts that can be derived from
them. Selected child plans are also available when a parent route is assessed.

The policy deliberately does not depend on leading batch dimensions, runtime
coefficient values, forward-versus-backward intent, or backend timing tables.
Those quantities can change the fastest implementation, but making them part
of `DefaultPolicy` would change the planning contract.

Feasibility is a separate question. `ResourceRequirements` first rejects
routes whose static storage or interaction requirements exceed the configured
budget. The policy compares only feasible candidates. A route is not preferred
because it uses fewer resources, and a larger resource budget does not make a
route faster.

Executor-internal micro-kernels are separate again. An executor may choose a
specialized determinant, partition a Taylor batch, or select a direct
high-grade action internally. Such choices may depend on runtime values or
placement. They are not promoted into planner-visible routes merely to improve
the score.

This gives three boundaries:

\[
\text{feasibility}
\quad\ne\quad
\text{route preference}
\quad\ne\quad
\text{executor-local dispatch}.
\]

## Work profiles before scores

A work profile is a ledger of structurally distinct work. Its coordinates are
chosen by reading the planner and executor together and asking two questions:

- what algorithmic work is actually performed by this route?
- which independent parts of that work are known when planning occurs?

Counts that are fixed multiples or linear combinations of the same structural
quantity are not turned into extra calibration dimensions. For example, two
gathers and one weighted term vector are all proportional to the same sparse
interaction count. Adding separate axes for each would only make the
calibration underdetermined.

The profile is therefore smaller than a FLOP accounting. It preserves
different scaling laws and different execution topologies, then leaves their
relative cost to calibration.

All counts below describe one broadcast item. Leading tensor dimensions,
allocation, dispatch, autograd retention, and other runtime effects are not
part of the structural basis.

## Product work

Let

\[
D=2^n
\]

be the full multivector width, \(O\) the requested compact output width, and
\(P\) the exact number of nonzero basis interactions retained by the grade
plan.

The sparse product gathers and evaluates those \(P\) interactions, then
reduces them to the output. A scalar output uses a contiguous reduction;
a wider output uses indexed accumulation. These have different execution
topologies, but their dominant interaction work is controlled by \(P\).

The full-table product gathers and weights a dense \(D\times D\) Cayley table
and contracts \(D\) terms into each of \(D\) outputs. Metric or
operation-specific zeros do not remove those dense table cells from the
executed path.

The minimal calibration basis is therefore

\[
\phi_{\mathrm{sparse}}=(P,O),
\qquad
\phi_{\mathrm{full}}=(D^2,D).
\]

Quantities such as \(2P\), \(3P\), \(D(D-1)\), or \(2D^2-D\) are useful when
auditing execution, but they add no independent calibration dimension.

The final shared score is

\[
S_{\mathrm{sparse}}=P+O,
\]

\[
S_{\mathrm{full}}=\tfrac12 D^2+32D.
\]

The coefficient 32 was not chosen as a hardware constant. The competitive
Product sample was small, and the selected route pattern stayed unchanged
over a broad output coefficient range. A simple representative from that
stable plateau was preferred to a more precise fitted decimal.

## Bivector exponential work

For a bivector exponential, define

\[
B=\binom n2,
\qquad
E=2^{n-1},
\]

where \(E\) is the width of the even subalgebra. Product children carry their
own Product work profiles into the exponential profile. For a selected Product
child \(P_c\), write the calibrated child charge as

\[
q(P_c)=P_c.\mathrm{bulk}+P_c.\mathrm{output}.
\]

### Closed forms

For \(n\le3\), the finite closure stays in the span of \(1\) and the
bivector. Its work contains the bivector-square reduction, scalar coefficient
functions, and projection to the requested output.

For \(4\le n\le5\), the closed route changes algorithmic regime. Writing

\[
B^2=s+K
\]

with grade-4 \(K\), the biquadratic closure additionally evaluates

\[
B\wedge B,\qquad K^2,\qquad BK,
\]

and combines the corresponding scalar, bivector, grade-4, and mixed terms.
The structural score retains those selected Product children instead of
collapsing them into one interaction total.

### Left-multiplication matrix

The matrix route constructs left multiplication by the bivector on the even
subalgebra. Its selected Product child is applied to each of the \(E\)
identity columns before an \(E\times E\) matrix exponential:

\[
S_{\mathrm{left}}
=
E\,q(P_{\mathrm{left}})
+
E^3
+
O.
\]

The \(E^3\) term is a structural matrix-order term, not a claim about the exact
kernel sequence used by `torch.matrix_exp`.

### Taylor evaluation

Taylor execution has two structurally different schedules. The plain schedule
uses the output-pruned Horner sequence. The scaled schedule evaluates a
full-even polynomial and then performs \(s\) even-even squarings:

\[
W_{\mathrm{scaled}}(s)
=
W_{\mathrm{scaled,base}}
+
sQ,
\]

where

\[
Q=q(P_{\mathrm{square}})+E,
\qquad
1\le s\le16.
\]

The actual branch and \(s\) depend on runtime coefficient norms, so
`DefaultPolicy` cannot use them. The static Taylor score instead uses both
known schedule envelopes without inspecting the input:

\[
S_{\mathrm{Taylor}}
=
\tfrac12 W_{\mathrm{plain}}
+
\tfrac12 W_{\mathrm{scaled,base}}
+
Q.
\]

This is a route-selection heuristic. It is not an assertion that a call
performs half of each schedule.

### The small-matrix boundary

A purely linear structural comparison initially failed at one concentrated
boundary. Fifteen of sixteen avoidable exponential mismatches occurred at

\[
n=4,\qquad E=8.
\]

At this dimension the closed route has just entered the three-child
biquadratic algorithm, while the matrix route still exponentiates only an
\(8\times8\) operator. The cubic structural term made the matrix route look
much more expensive than it behaved in this small regime.

The mismatch also followed a planner-visible numerical distinction. The
closed biquadratic evaluator enables additional divided-difference
stabilization when

\[
r>0
\quad\text{or}\quad
(p>0\ \text{and}\ q>0),
\]

and float32 has its own precision-dependent crossover. Pure Euclidean float64
at the same dimension was an important contrasting case where the shared
oracle retained the closed route.

The final policy therefore treats this as an explicit algorithmic regime:

- if \(E=8\), and
- dtype is float32 or the signature activates the stabilized closed path,

then a feasible `left_matrix_exp` route is preferred.

Outside that window the ordinary structural comparison is used. This is
preferable to encoding the result as a fictitious claim that the matrix route
performs some tiny fraction of its structural work. The boundary is attached
to a real change in the closed algorithm and a genuinely small matrix order.

## Action work

Versor actions have three competing algorithmic families:

- build a vector-space transformation and lift it to requested grades;
- materialize a rotor and apply two Clifford products;
- materialize a full sandwich-action matrix.

The structural model preserves those differences rather than reducing every
route to a common interaction count.

### Lifting a vector-space map

For a shared grade \(g\), let

\[
C_g=\binom ng^2
\]

be the number of coefficients in its exterior-power block. The compound path
computes minors of the vector matrix. Its determinant work is modeled by

\[
C_g d_g,
\]

with the explicit small-determinant costs used by the implementation and a
cubic order for higher grades.

The direct action instead propagates exterior-power states through transition
terms. Let

\[
T_g
=
\sum_{j=0}^{g-1}
\binom n{j+1}
\binom n{g-j-1}
(n-g+j+1)
\]

count those transition terms, and

\[
U_g
=
\sum_{j=0}^{g-1}
\binom n{j+1}
\binom n{g-j-1}
(n-g+j)
\]

count the corresponding reduction additions.

The structural representation retains separate compound, direct, and
hybrid-\(S\) equations. It does not claim that the executor always chooses the
cheapest structural one. The actual direct-grade set can depend on device and
layout heuristics and remains executor-local.

For the device-independent policy score, the calibrated direct endpoint is
used as a stable route descriptor rather than inspecting that actual
executor-local set.

### Composite action score

The final Action score is a small positive combination of the accepted
profile components. Absent components contribute zero. In normalized form it
uses:

- generator and reflection work with unit weight;
- vector-space matrix-exponential order with coefficient \(0.2\);
- selected Product-child bulk work with coefficient \(100\);
- selected Product-child output work with unit weight;
- the selected exponential child's static score with unit weight;
- the device-independent direct-lift endpoint with coefficient \(20\);
- the full-action matrix and retained linear/output terms from the route
  profile.

The large difference between Product-child bulk and output coefficients is a
calibrated route-comparison weight, not a statement that one arithmetic
operation is literally one hundred times another. The matrix coefficient was
stable over roughly \(0.1\) to \(0.3\); the lift coefficient had a narrower
stable region around \(20\) to \(21\). Simple representatives were chosen from
those plateaus.

## Forced-route measurements

The structural basis was derived before using timing data. Benchmark
artifacts were then used in two different roles.

First, they were used as a **sanity check**. Every feasible root route was
forced for the same semantic requests, allowing route timings to be compared
without changing the requested mathematics. The measurements covered six
placements:

- Apple M5 Pro CPU float32;
- Apple M5 Pro CPU float64;
- Apple M5 Pro MPS float32;
- AMD EPYC CPU float32;
- AMD EPYC CPU float64;
- NVIDIA CUDA float32.

Steady forward and forward-plus-backward measurements were both retained.
They are different hidden execution conditions for the same static planning
request, not different policy inputs.

For this first diagnostic, coefficients were temporarily allowed to differ by
environment and timing mode. That fit was intentionally optimistic. Its
purpose was only to ask whether the structural coordinates had enough
expressive power to reproduce the observed crossover shapes. Failure even
under those relaxed conditions would have sent the work model back to the
algorithm audit.

## Separating structural error from missing information

Even a perfect structural score cannot reproduce a route choice that depends
on information hidden from the planner. The calibration therefore used three
oracle levels.

For a measured row \(x\), route \(r\), and timing \(T(x,r)\), define
multiplicative regret

\[
q(x,r)
=
\frac{T(x,r)}{\min_{r'}T(x,r')},
\qquad
q\ge1.
\]

The optimization uses log regret,

\[
\ell(x,r)=\log q(x,r),
\]

so minimizing mean log regret is equivalent to minimizing geometric-mean
multiplicative regret.

The **pointwise oracle** chooses the fastest route for every measured row. Its
regret is one by definition, but it can use batch shape, runtime values,
timing mode, device, and every other hidden condition.

The **environment-static oracle** requires one route for all rows with the
same planner-visible context inside one environment.

The **shared static oracle** adds the final design constraint: the same static
context must choose the same route across all six environments. Device is not
part of its key.

The static context contains the information available to planning, such as
signature, operation, declared layouts, action grade, and dtype. It excludes
leading shape, generator values, Taylor branch and square count, timing mode,
and executor-local micro-kernel choices.

The gap between the pointwise and static oracles measures unavoidable
information loss. The gap between a structural score and the shared static
oracle measures the part the scoring model could still improve.

This distinction was particularly important for the exponential family,
where many apparent winner conflicts disappeared only after hidden generator
structure or batch shape was fixed. Those conflicts were not evidence for a
new planner-visible Fact.

## Context-balanced calibration

The benchmark suite is a coverage instrument, not a model of production
traffic. Some structural contexts contain more batch shapes or generated
inputs than others. Treating every row as an independent training sample would
therefore give accidental extra weight to heavily enumerated contexts.

Final calibration gave equal weight to each planner-visible static context.
Rows inside a context were treated as hidden conditions that the same route
must survive.

A small positive finite search was sufficient. Coefficients were searched on
coarse grids, with at most a local refinement near useful regions. Nearby
choices were inspected for stable plateaus. The objective was not to recover a
high-precision numerical optimum, but to find a simple shared rule whose
decisions remained stable under small coefficient changes.

Route-order accuracy was recorded as a diagnostic but was not optimized
directly. Choosing the slower route in a near-tie matters much less than
choosing a route that is ten times slower. Regret captures that distinction.

Tail regret was inspected alongside the mean. This exposed the avoidable
\(n=4\) exponential failure that a good aggregate score alone would have
hidden.

## Final calibration

The final production policy reproduced the accepted shared candidate behavior:

| Family | Context-balanced geometric regret | Row p90 | Row maximum |
| --- | ---: | ---: | ---: |
| Product | 1.0626x | 1.3426x | 2.6486x |
| BivectorExp | 1.2727x | 3.0267x | 11.7762x |
| Action | 1.0343x | 1.0843x | 2.1701x |

Product was effectively at its shared-static lower bound. Action remained
close to its shared-static oracle with a small residual cost for using one
device-independent score.

For BivectorExp, the explicit \(E=8\) regime removed the avoidable CUDA
closed-versus-matrix tail and brought context-balanced regret essentially to
the shared-static oracle. The remaining roughly \(11.8\times\) pointwise tail
is also incurred by the shared static oracle: it comes from a hidden
backend-specific preference that a single device-independent static policy
cannot resolve.

That remaining tail is therefore qualitatively different from the removed
regression. Eliminating it would require changing the policy information
boundary or introducing device-specific selection, not merely finding a
better coefficient.

## What the model claims

The structural model claims that route selection is based on algorithmic work
that can be known when planning occurs, with a small empirical calibration of
their relative importance.

It does not claim that the score predicts latency, that its coefficients are
portable hardware constants, or that every runtime call follows the exact
structural path used by the score. It also does not use benchmark observations
to invent arbitrary planner features.

The methodology is:

\[
\text{executor algorithms}
\rightarrow
\text{planner-visible structural work}
\rightarrow
\text{information-boundary audit}
\rightarrow
\text{forced-route diagnostics}
\rightarrow
\text{static-oracle lower bounds}
\rightarrow
\text{small shared calibration}
\rightarrow
\text{DefaultPolicy}.
\]

The important ordering is that benchmark data enters after the structural
equations have been derived. Measurements calibrate crossovers between known
algorithmic regimes; they do not replace the model of those regimes.
