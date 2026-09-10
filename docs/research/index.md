# Transformation Fields

`research/transformation_fields` explores differentiable fields of local
Clifford-generated transformations. The field machinery stays outside clifra's
stable core: sampling, interpolation, objectives, and application structure are
ordinary PyTorch code around planned Clifford actions.

For a bivector generator \(B\),

\[
R(t)=\exp(-tB/2), \qquad
x(t)=R(t)x(0)\widetilde{R(t)}.
\]

Equivalently, on the requested representation,

\[
x(1)=\exp(G(B))x(0).
\]

A transformation field makes the generator depend on a sample identity
\(\xi\). The value being transformed and the identity selecting its generator
need not be the same object:

\[
\phi_\Theta(x,\xi)
=
\exp(G(B_{S-1}^\Theta(\xi)))\cdots
\exp(G(B_0^\Theta(\xi)))x.
\]

This distinction lets a material coordinate, acquisition time, or spatial
position remain attached to a sample while the transformed value changes. For
fixed identities, reversing the ordered actions and negating their generators
gives the indexed inverse.

## Experiments

The examples use the same basic construction in three different geometries.

### [Sparse-Constraint Continuum Threading](sparse_continuum_threading.md)

A material-coordinate field in \(Cl(4,1)\) is fitted from three gate poses and
one tip pose. The optimized field is then evaluated on a much denser continuum
discretization without retraining.

### [Continuous-Time PGA LiDAR Deskewing](pga-lidar-deskewing.md)

A motor field in \(Cl(3,0,1)\) uses acquisition time as persistent identity.
Sparse point-plane, point-line, and anchor constraints recover continuous
ego-motion and deskew an independently sampled scan.

### [Relativistic Beamline Inverse Design](relativistic-beamline.md)

A \(Cl(1,3)\) electromagnetic bivector field steers a relativistic particle
bunch through offset apertures toward a target. Lorentz rotor updates preserve
the four-velocity mass shell while the field is optimized end to end.
