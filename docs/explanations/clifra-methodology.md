# Geometric Parameterization

Clifra is a set of Clifford algebra tools for PyTorch. It does not prescribe a
model architecture, training objective, or scientific domain. One supported
method is to represent a geometric object's generating coordinates directly and
construct its action with the algebra.

Under this method, a model learns a bivector that generates a rotor, a vector
that determines a reflection, or coefficients in a declared mixture of grades.
The learned quantity is the geometric generator rather than an unconstrained
tensor fitted to the resulting transformation.

`VersorLayer` and `MultiVersorLayer` currently support `grade=1` reflection
parameters and `grade=2` rotor parameters. Other grade layouts remain available
to planned products and project-specific layers; they are not accepted as
versor-layer parameter grades.

## From coordinates to an action

A typical construction has four stages:

1. **Declare an algebra.** The signature states how basis directions square and
   therefore which geometry the products express.
2. **Declare a layout.** The selected grades define the coordinate space of the
   object being learned.
3. **Generate an action.** Clifford products, exponentials, reverses, and
   sandwich actions turn those coordinates into a transformation.
4. **Learn through the action.** PyTorch autograd differentiates the loss through
   the planned tensor program and back to the coordinates.

For a bivector $B$, a rotor can be written

\[
R = \exp(-B/2), \qquad x' = R x \widetilde{R}.
\]

The parameter is $B$, rather than an arbitrary dense matrix or a stored rotor
whose constraints must be repaired after every update. The exponential and
sandwich product construct the action at each forward pass. The model therefore
learns the plane generator of the transformation in a fixed grade-2 coordinate
space.

The parameterization supplies no performance guarantee. It applies when the
chosen signature, grades, and action match the problem structure; otherwise it
can impose an unnecessary restriction.

## Layout is part of the hypothesis

A layout does more than reduce storage. Selecting grade 1 places the
coefficients in the vector subspace of the chosen algebra. Selecting grade 2
places them in the bivector subspace of oriented plane elements. Selecting
several grades permits a mixed-grade multivector object. A transformation law
depends on the operation or layer applied to that layout.

The layout is consequently part of the model hypothesis. Narrowing it can make
the representation interpretable and computationally tractable, but it also
excludes components. Clifra leaves this choice visible instead of silently
embedding every object in a full $2^n$-lane multivector.

## The continuum solver is one local example

The continuum solver demonstrates the proposal without defining it. Its
`CoordinateChart` embeds ordinary coordinates into a declared grade-1 space and
extracts them again. An `InvertibleBivectorField` learns a sequence or control
lattice of grade-2 generators. Planned versor actions exponentiate and apply
those generators along a path; reversing their order and sign constructs the
inverse path.

The learnable object is therefore the transformation field itself, expressed by
bivector coordinates. Ordinary PyTorch optimization updates those coordinates
through losses evaluated on the transformed points.

The solver also shows what the algebra does not supply. Target criteria,
sampling, curriculum, path-consistency checks, boundary conditions, and other
domain constraints are application policy. They are injected by the solver or
training program. Clifford structure can make an action invertible by
construction while still leaving the learned field scientifically unsuitable.

## Other applications

Applications can include:

- geometric deep-learning layers whose parameters are blades or versors;
- learned coordinate systems and features designed for equivariance or
  invariance;
- fields assembled from local geometric generators;
- research on alternative signatures, layouts, actions, or planning policies;
- direct Clifford algebra computation inside a larger scientific program.

None of these uses requires adopting the continuum solver. Clifra's reusable
foundation is the algebra specification, layout contracts, static planners,
tensor executors, and differentiable operations. A project may use one layer,
build a new family of geometric modules, or use clifra only for computation.

## Application responsibilities

Learning a geometric object does not establish that it is the correct object.
The application must still determine:

- whether the signature represents the intended geometry;
- whether the selected grades are sufficient;
- which action should be generated from the parameters;
- which invariants must hold for the whole model rather than one layer;
- how numerical approximation affects the intended regime;
- which data and objective identify the desired solution.

Clifra exposes these decisions and supplies algebraic structure after they are
made. It does not enforce them or classify every tensor as geometric.
