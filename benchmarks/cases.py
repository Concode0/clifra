"""Declarative semantic cases and execution placements for three routed families."""

from __future__ import annotations

from dataclasses import dataclass

PRODUCT_OPERATIONS = {
    "geometric_product",
    "wedge",
    "left_contraction",
    "right_contraction",
    "symmetric_product",
    "commutator_product",
    "anti_commutator_product",
}
DTYPES = {"float32", "float64"}
DEVICES = {"cpu", "cuda", "mps"}
STORAGE_MODES = {"compact", "canonical"}
FAMILIES = {"product", "bivector_exp", "action"}
ACTION_OPERATIONS = {"linear", "versor", "sandwich"}
GENERATOR_STRUCTURES = {
    "random",
    "simple_plane",
    "commuting_planes",
    "null_plane",
    "near_identity_matrix",
    "reflection_normal",
}
DENSE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class LayoutCase:
    """One operand's semantic grades, physical storage, and ordinary shape."""

    grades: tuple[int, ...]
    storage: str = "compact"
    leading_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "grades", tuple(self.grades))
        object.__setattr__(self, "leading_shape", tuple(self.leading_shape))
        if self.storage not in STORAGE_MODES:
            raise ValueError(f"unsupported storage mode {self.storage!r}")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.grades):
            raise ValueError("grades must be non-negative integers")
        if tuple(sorted(set(self.grades))) != self.grades:
            raise ValueError("grades must be sorted and unique")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in self.leading_shape):
            raise ValueError("ordinary leading dimensions must be positive integers")

    def to_dict(self) -> dict[str, object]:
        return {
            "grades": list(self.grades),
            "storage": self.storage,
            "leading_shape": list(self.leading_shape),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> LayoutCase:
        return cls(tuple(value["grades"]), str(value.get("storage", "compact")), tuple(value.get("leading_shape", ())))


@dataclass(frozen=True)
class OrdinaryTensorCase:
    """One non-Clifford tensor argument, with explicit leading and trailing shape."""

    leading_shape: tuple[int, ...] = ()
    trailing_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "leading_shape", tuple(self.leading_shape))
        object.__setattr__(self, "trailing_shape", tuple(self.trailing_shape))
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (*self.leading_shape, *self.trailing_shape)
        ):
            raise ValueError("ordinary tensor dimensions must be positive integers")

    def to_dict(self) -> dict[str, object]:
        return {"leading_shape": list(self.leading_shape), "trailing_shape": list(self.trailing_shape)}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> OrdinaryTensorCase:
        return cls(tuple(value.get("leading_shape", ())), tuple(value.get("trailing_shape", ())))


@dataclass(frozen=True)
class SelectionCase:
    """Normal policy selection, or an explicitly benchmark-private forced route."""

    mode: str = "default"
    route: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"default", "forced_repository_private"}:
            raise ValueError(f"unsupported selection mode {self.mode!r}")
        if self.mode == "default" and self.route is not None:
            raise ValueError("default selection must not request a route")
        if self.mode == "forced_repository_private" and not self.route:
            raise ValueError("forced selection requires a route")

    def to_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "requested_route": self.route}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> SelectionCase:
        return cls(str(value.get("mode", "default")), value.get("requested_route"))


@dataclass(frozen=True)
class ExecutionPlacement:
    """Execution metadata deliberately excluded from semantic case identity."""

    dtype: str = "float32"
    device: str = "cpu"
    selection: SelectionCase = SelectionCase()

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        if self.device not in DEVICES:
            raise ValueError(f"unsupported device {self.device!r}")

    def to_dict(self) -> dict[str, object]:
        return {"dtype": self.dtype, "device": self.device, "selection": self.selection.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ExecutionPlacement:
        return cls(
            str(value.get("dtype", "float32")),
            str(value.get("device", "cpu")),
            SelectionCase.from_dict(value.get("selection", {})),
        )


@dataclass(frozen=True)
class BenchmarkCase:
    """A fixed mathematical request, independent of execution placement.

    The stable ``case_id`` is supplied by the author rather than derived from
    display names or implementation details.
    """

    case_id: str
    operation: str
    signature: tuple[int, int, int]
    inputs: tuple[LayoutCase, ...]
    output: LayoutCase
    seed: int = 1729
    family: str = "product"
    ordinary_inputs: tuple[OrdinaryTensorCase, ...] = ()
    action_grade: int | None = None
    generator_structure: str = "random"
    value_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "signature", tuple(self.signature))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "ordinary_inputs", tuple(self.ordinary_inputs))
        if not self.case_id or any(character.isspace() for character in self.case_id):
            raise ValueError("case_id must be a non-empty stable token without whitespace")
        if self.family not in FAMILIES:
            raise ValueError(f"unsupported benchmark family {self.family!r}")
        if len(self.signature) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.signature
        ):
            raise ValueError("signature must contain three non-negative integers")
        if self.family == "product" and (self.operation not in PRODUCT_OPERATIONS or len(self.inputs) != 2):
            raise ValueError("product cases require a supported operation and exactly two Clifford inputs")
        if self.family == "bivector_exp" and (self.operation != "bivector_exp" or len(self.inputs) != 1):
            raise ValueError("bivector_exp cases require exactly one Clifford input")
        if self.family == "action" and self.operation not in ACTION_OPERATIONS:
            raise ValueError(f"unsupported routed action operation {self.operation!r}")
        if self.family == "action" and self.operation == "linear":
            if len(self.inputs) != 1 or len(self.ordinary_inputs) != 1 or self.action_grade is not None:
                raise ValueError("linear action cases require one Clifford value and one ordinary matrix")
            n = sum(self.signature)
            if self.ordinary_inputs[0].trailing_shape != (n, n):
                raise ValueError(f"linear action matrix trailing shape must be {(n, n)}")
        if self.family == "action" and self.operation == "versor":
            valid_grade = (
                not isinstance(self.action_grade, bool)
                and isinstance(self.action_grade, int)
                and self.action_grade in {1, 2}
            )
            if len(self.inputs) != 2 or self.ordinary_inputs or not valid_grade:
                raise ValueError("versor action cases require values, a grade-1/2 parameter, and action_grade")
            if self.inputs[1].grades != (self.action_grade,):
                raise ValueError("versor parameter grades must match action_grade")
        if self.family == "action" and self.operation == "sandwich":
            if len(self.inputs) != 3 or self.ordinary_inputs or self.action_grade is not None:
                raise ValueError("sandwich action cases require left, input, and right Clifford values")
        if self.family != "action" and (self.ordinary_inputs or self.action_grade is not None):
            raise ValueError("ordinary inputs and action_grade are action-only declarations")
        if self.generator_structure not in GENERATOR_STRUCTURES:
            raise ValueError(f"unsupported generator structure {self.generator_structure!r}")
        if self.family == "bivector_exp" and self.inputs[0].grades != (2,):
            raise ValueError("bivector_exp benchmark inputs must declare grade 2")
        if not isinstance(self.value_scale, (int, float)) or isinstance(self.value_scale, bool) or self.value_scale < 0:
            raise ValueError("value_scale must be a non-negative number")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        n = sum(self.signature)
        for declaration in (*self.inputs, self.output):
            if declaration.grades and declaration.grades[-1] > n:
                raise ValueError(f"grade exceeds algebra dimension {n}")

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "operation": self.operation,
            "signature": list(self.signature),
            "inputs": [item.to_dict() for item in self.inputs],
            "output": self.output.to_dict(),
            "seed": self.seed,
            "ordinary_inputs": [item.to_dict() for item in self.ordinary_inputs],
            "action_grade": self.action_grade,
            "generator_structure": self.generator_structure,
            "value_scale": self.value_scale,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> BenchmarkCase:
        return cls(
            case_id=str(value["case_id"]),
            family=str(value.get("family", "product")),
            operation=str(value["operation"]),
            signature=tuple(value["signature"]),
            inputs=tuple(LayoutCase.from_dict(item) for item in value["inputs"]),
            output=LayoutCase.from_dict(value["output"]),
            seed=int(value.get("seed", 1729)),
            ordinary_inputs=tuple(OrdinaryTensorCase.from_dict(item) for item in value.get("ordinary_inputs", ())),
            action_grade=value.get("action_grade"),
            generator_structure=str(value.get("generator_structure", "random")),
            value_scale=float(value.get("value_scale", 1.0)),
        )


def smoke_cases() -> tuple[BenchmarkCase, ...]:
    """Return tiny representative semantic cases for all routed families."""

    return (
        BenchmarkCase(
            case_id="product.cl3.full.gp.batch4",
            operation="geometric_product",
            signature=(3, 0, 0),
            inputs=(LayoutCase((0, 1, 2, 3), leading_shape=(4,)),) * 2,
            output=LayoutCase((0, 1, 2, 3)),
        ),
        BenchmarkCase(
            case_id="product.cl21.vector-vector.wedge.broadcast3x2",
            operation="wedge",
            signature=(2, 1, 0),
            inputs=(LayoutCase((1,), leading_shape=(3, 1)), LayoutCase((1,), leading_shape=(1, 2))),
            output=LayoutCase((2,)),
            seed=1730,
        ),
        BenchmarkCase(
            case_id="bivector_exp.cl3.even.simple-plane.batch2",
            family="bivector_exp",
            operation="bivector_exp",
            signature=(3, 0, 0),
            inputs=(LayoutCase((2,), leading_shape=(2,)),),
            output=LayoutCase((0, 2)),
            generator_structure="simple_plane",
            value_scale=0.2,
            seed=1731,
        ),
        BenchmarkCase(
            case_id="action.cl3.vector.rotor.broadcast2x3",
            family="action",
            operation="versor",
            signature=(3, 0, 0),
            inputs=(LayoutCase((1,), leading_shape=(2, 1)), LayoutCase((2,), leading_shape=(1, 3))),
            output=LayoutCase((1,)),
            action_grade=2,
            generator_structure="simple_plane",
            value_scale=0.15,
            seed=1732,
        ),
        BenchmarkCase(
            case_id="action.cl3.full.rotor.batch2",
            family="action",
            operation="versor",
            signature=(3, 0, 0),
            inputs=(LayoutCase((0, 1, 2, 3), leading_shape=(2,)), LayoutCase((2,), leading_shape=(2,))),
            output=LayoutCase((0, 1, 2, 3)),
            action_grade=2,
            generator_structure="commuting_planes",
            value_scale=0.12,
            seed=1733,
        ),
        BenchmarkCase(
            case_id="action.cl3.grade2.linear.broadcast2x3",
            family="action",
            operation="linear",
            signature=(3, 0, 0),
            inputs=(LayoutCase((2,), leading_shape=(2, 1)),),
            ordinary_inputs=(OrdinaryTensorCase((1, 3), (3, 3)),),
            output=LayoutCase((2,)),
            generator_structure="near_identity_matrix",
            value_scale=0.1,
            seed=1734,
        ),
        BenchmarkCase(
            case_id="action.cl3.full.sandwich.shared-pair.batch2",
            family="action",
            operation="sandwich",
            signature=(3, 0, 0),
            inputs=(
                LayoutCase((0, 1, 2, 3)),
                LayoutCase((0, 1, 2, 3), leading_shape=(2,)),
                LayoutCase((0, 1, 2, 3)),
            ),
            output=LayoutCase((0, 1, 2, 3)),
            value_scale=0.1,
            seed=1735,
        ),
    )


def _product_case(
    case_id: str,
    operation: str,
    signature: tuple[int, int, int],
    left_grades: tuple[int, ...],
    right_grades: tuple[int, ...],
    output_grades: tuple[int, ...],
    *,
    leading_shapes: tuple[tuple[int, ...], tuple[int, ...]],
    storage: tuple[str, str, str] = ("compact", "compact", "compact"),
    seed: int,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        operation=operation,
        signature=signature,
        inputs=(
            LayoutCase(left_grades, storage[0], leading_shapes[0]),
            LayoutCase(right_grades, storage[1], leading_shapes[1]),
        ),
        output=LayoutCase(output_grades, storage[2]),
        seed=seed,
    )


def _exploratory_workloads() -> tuple[dict[str, object], ...]:
    """Purposeful product workloads; dtype, device, and route are placements."""

    workloads: list[dict[str, object]] = []
    for batch in DENSE_BATCHES:
        workloads.append(
            {
                "case_id": f"product.cl5.full-full.gp.batch{batch}",
                "operation": "geometric_product",
                "signature": (5, 0, 0),
                "left_grades": tuple(range(6)),
                "right_grades": tuple(range(6)),
                "output_grades": tuple(range(6)),
                "leading_shapes": ((batch,), (batch,)),
            }
        )
    for batch in (1, 64, 1024, 4096):
        workloads.append(
            {
                "case_id": f"product.cl12.vector-bivector.gp.batch{batch}",
                "operation": "geometric_product",
                "signature": (12, 0, 0),
                "left_grades": (1,),
                "right_grades": (2,),
                "output_grades": (1, 3),
                "leading_shapes": ((batch,), (batch,)),
            }
        )
    workloads.extend(
        [
            {
                "case_id": "product.cl21.vector-vector.gp.batch64",
                "operation": "geometric_product",
                "signature": (2, 1, 0),
                "left_grades": (1,),
                "right_grades": (1,),
                "output_grades": (0, 2),
                "leading_shapes": ((64,), (64,)),
            },
            {
                "case_id": "product.cl301.vector-vector.wedge.canonical.batch64",
                "operation": "wedge",
                "signature": (3, 0, 1),
                "left_grades": (1,),
                "right_grades": (1,),
                "output_grades": (2,),
                "leading_shapes": ((64,), (64,)),
                "storage": ("canonical", "canonical", "canonical"),
            },
            {
                "case_id": "product.cl32.vector-bivector.gp.batch256",
                "operation": "geometric_product",
                "signature": (3, 2, 0),
                "left_grades": (1,),
                "right_grades": (2,),
                "output_grades": (1, 3),
                "leading_shapes": ((256,), (256,)),
            },
            {
                "case_id": "product.cl41.bivector-vector.right-contraction.batch256",
                "operation": "right_contraction",
                "signature": (4, 1, 0),
                "left_grades": (2,),
                "right_grades": (1,),
                "output_grades": (1,),
                "leading_shapes": ((256,), (256,)),
            },
            {
                "case_id": "product.cl33.bivector-bivector.gp.batch128",
                "operation": "geometric_product",
                "signature": (3, 3, 0),
                "left_grades": (2,),
                "right_grades": (2,),
                "output_grades": (0, 2, 4),
                "leading_shapes": ((128,), (128,)),
            },
            {
                "case_id": "product.cl501.bivector-bivector.left-contraction.batch128",
                "operation": "left_contraction",
                "signature": (5, 0, 1),
                "left_grades": (2,),
                "right_grades": (2,),
                "output_grades": (0,),
                "leading_shapes": ((128,), (128,)),
            },
            {
                "case_id": "product.cl8.vector-trivector.wedge.batch64",
                "operation": "wedge",
                "signature": (8, 0, 0),
                "left_grades": (1,),
                "right_grades": (3,),
                "output_grades": (4,),
                "leading_shapes": ((64,), (64,)),
            },
            {
                "case_id": "product.cl44.bivector-trivector.gp.broadcast32x8",
                "operation": "geometric_product",
                "signature": (4, 4, 0),
                "left_grades": (2,),
                "right_grades": (3,),
                "output_grades": (1, 3, 5),
                "leading_shapes": ((32, 1), (1, 8)),
            },
            {
                "case_id": "product.cl10.trivector-trivector.wedge.batch16",
                "operation": "wedge",
                "signature": (10, 0, 0),
                "left_grades": (3,),
                "right_grades": (3,),
                "output_grades": (6,),
                "leading_shapes": ((16,), (16,)),
            },
            {
                "case_id": "product.cl66.bivector-bivector.commutator.batch64",
                "operation": "commutator_product",
                "signature": (6, 6, 0),
                "left_grades": (2,),
                "right_grades": (2,),
                "output_grades": (2,),
                "leading_shapes": ((64,), (64,)),
            },
            {
                "case_id": "product.cl12.trivector-trivector.gp.batch8",
                "operation": "geometric_product",
                "signature": (12, 0, 0),
                "left_grades": (3,),
                "right_grades": (3,),
                "output_grades": (0, 2, 4, 6),
                "leading_shapes": ((8,), (8,)),
            },
            {
                "case_id": "product.cl24.vector-bivector.gp.batch16",
                "operation": "geometric_product",
                "signature": (24, 0, 0),
                "left_grades": (1,),
                "right_grades": (2,),
                "output_grades": (1, 3),
                "leading_shapes": ((16,), (16,)),
            },
            {
                "case_id": "product.cl6.even-even.gp.batch128",
                "operation": "geometric_product",
                "signature": (6, 0, 0),
                "left_grades": (0, 2, 4, 6),
                "right_grades": (0, 2, 4, 6),
                "output_grades": (0, 2, 4, 6),
                "leading_shapes": ((128,), (128,)),
            },
            {
                "case_id": "product.cl71.mixed-mixed.symmetric.broadcast32x8",
                "operation": "symmetric_product",
                "signature": (7, 1, 0),
                "left_grades": (0, 1, 2),
                "right_grades": (1, 3),
                "output_grades": tuple(range(9)),
                "leading_shapes": ((32, 1), (1, 8)),
            },
            {
                "case_id": "product.cl22.full-full.commutator.batch256",
                "operation": "commutator_product",
                "signature": (2, 2, 0),
                "left_grades": tuple(range(5)),
                "right_grades": tuple(range(5)),
                "output_grades": tuple(range(5)),
                "leading_shapes": ((256,), (256,)),
            },
            {
                "case_id": "product.cl401.full-full.anti-commutator.batch128",
                "operation": "anti_commutator_product",
                "signature": (4, 0, 1),
                "left_grades": tuple(range(6)),
                "right_grades": tuple(range(6)),
                "output_grades": tuple(range(6)),
                "leading_shapes": ((128,), (128,)),
            },
            {
                "case_id": "product.cl6.full-full.wedge.batch64",
                "operation": "wedge",
                "signature": (6, 0, 0),
                "left_grades": tuple(range(7)),
                "right_grades": tuple(range(7)),
                "output_grades": tuple(range(7)),
                "leading_shapes": ((64,), (64,)),
            },
            {
                "case_id": "product.cl8.full-full.gp.batch16",
                "operation": "geometric_product",
                "signature": (8, 0, 0),
                "left_grades": tuple(range(9)),
                "right_grades": tuple(range(9)),
                "output_grades": tuple(range(9)),
                "leading_shapes": ((16,), (16,)),
            },
            {
                "case_id": "product.cl32.mixed-mixed.gp.canonical.batch32",
                "operation": "geometric_product",
                "signature": (3, 2, 0),
                "left_grades": (0, 1, 2),
                "right_grades": (1, 2, 3),
                "output_grades": tuple(range(6)),
                "leading_shapes": ((32,), (32,)),
                "storage": ("canonical", "canonical", "canonical"),
            },
        ]
    )
    return tuple(workloads)


def exploratory_product_cases() -> tuple[BenchmarkCase, ...]:
    """Return the existing broad product workload matrix as semantic cases."""

    return tuple(
        _product_case(**workload, seed=2000 + seed_offset)
        for seed_offset, workload in enumerate(_exploratory_workloads())
    )


def _signature_tracks(dimensions: tuple[int, ...]) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    tracks = []
    for n in dimensions:
        tracks.append(("euclidean", (n, 0, 0)))
        if n >= 3:
            tracks.extend((("mixed", (n - 1, 1, 0)), ("null", (n - 1, 0, 1))))
    return tuple(tracks)


def exploratory_bivector_exp_cases() -> tuple[BenchmarkCase, ...]:
    """Generate a broad valid exp matrix from signature, output, batch, and generator tracks."""

    cases = []
    seed = 3000
    for domain, signature in _signature_tracks((2, 3, 4, 5, 6, 8, 10, 12)):
        n = sum(signature)
        outputs = (("scalar", (0,)), ("even", tuple(range(0, n + 1, 2))), ("full", tuple(range(n + 1))))
        structures = ("simple_plane", "random")
        if n >= 4:
            structures += ("commuting_planes",)
        if domain == "null":
            structures += ("null_plane",)
        for output_name, output_grades in outputs:
            for batch in (1, 32, 512):
                for structure in structures:
                    case_id = f"bivector_exp.cl{signature[0]}{signature[1]}{signature[2]}.{output_name}.{structure}.batch{batch}"
                    cases.append(
                        BenchmarkCase(
                            case_id=case_id,
                            family="bivector_exp",
                            operation="bivector_exp",
                            signature=signature,
                            inputs=(LayoutCase((2,), leading_shape=(batch,)),),
                            output=LayoutCase(output_grades),
                            generator_structure=structure,
                            value_scale=0.12,
                            seed=seed,
                        )
                    )
                    seed += 1
    return tuple(cases)


def _action_layout_tracks(n: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    tracks = [("vector", (1,)), ("bivector", (2,))]
    if n >= 3:
        tracks.append(("higher", (3,)))
    if n <= 6:
        tracks.append(("full", tuple(range(n + 1))))
    return tuple(tracks)


def exploratory_action_cases() -> tuple[BenchmarkCase, ...]:
    """Generate routed linear, rotor, and reflection-like action workloads."""

    cases = []
    seed = 4000
    broadcast_tracks = (
        ("aligned", (32,), (32,)),
        ("one-generator", (512,), ()),
        ("outer", (32, 1), (1, 8)),
    )
    for _domain, signature in _signature_tracks((3, 4, 5, 6, 8, 10, 12)):
        n = sum(signature)
        for layout_name, grades in _action_layout_tracks(n):
            for broadcast_name, value_shape, parameter_shape in broadcast_tracks:
                cases.append(
                    BenchmarkCase(
                        case_id=(
                            f"action.cl{signature[0]}{signature[1]}{signature[2]}.{layout_name}.rotor.{broadcast_name}"
                        ),
                        family="action",
                        operation="versor",
                        signature=signature,
                        inputs=(
                            LayoutCase(grades, leading_shape=value_shape),
                            LayoutCase((2,), leading_shape=parameter_shape),
                        ),
                        output=LayoutCase(grades),
                        action_grade=2,
                        generator_structure="commuting_planes" if n >= 4 else "simple_plane",
                        value_scale=0.1,
                        seed=seed,
                    )
                )
                seed += 1
            cases.append(
                BenchmarkCase(
                    case_id=f"action.cl{signature[0]}{signature[1]}{signature[2]}.{layout_name}.reflection.aligned",
                    family="action",
                    operation="versor",
                    signature=signature,
                    inputs=(LayoutCase(grades, leading_shape=(32,)), LayoutCase((1,), leading_shape=(32,))),
                    output=LayoutCase(grades),
                    action_grade=1,
                    generator_structure="reflection_normal",
                    value_scale=0.2,
                    seed=seed,
                )
            )
            seed += 1

        for layout_name, grades in _action_layout_tracks(n):
            # Full grade lifts beyond n=6 are deliberately absent from the
            # layout track; their determinant and coefficient matrix sizes are
            # not useful under the repository's default resource envelope.
            for broadcast_name, value_shape, matrix_shape in broadcast_tracks:
                cases.append(
                    BenchmarkCase(
                        case_id=(
                            f"action.cl{signature[0]}{signature[1]}{signature[2]}.{layout_name}.linear.{broadcast_name}"
                        ),
                        family="action",
                        operation="linear",
                        signature=signature,
                        inputs=(LayoutCase(grades, leading_shape=value_shape),),
                        ordinary_inputs=(OrdinaryTensorCase(matrix_shape, (n, n)),),
                        output=LayoutCase(grades),
                        generator_structure="near_identity_matrix",
                        value_scale=0.08,
                        seed=seed,
                    )
                )
                seed += 1
    cases.extend(
        (
            BenchmarkCase(
                case_id="action.cl300.full.sandwich.aligned",
                family="action",
                operation="sandwich",
                signature=(3, 0, 0),
                inputs=(LayoutCase(tuple(range(4)), leading_shape=(32,)),) * 3,
                output=LayoutCase(tuple(range(4))),
                value_scale=0.1,
                seed=seed,
            ),
            BenchmarkCase(
                case_id="action.cl500.generic.sandwich.broadcast-reuse32x8",
                family="action",
                operation="sandwich",
                signature=(5, 0, 0),
                inputs=(
                    LayoutCase((0, 2, 4), leading_shape=(1, 8)),
                    LayoutCase((1,), leading_shape=(32, 8)),
                    LayoutCase((0, 2, 4), leading_shape=(1, 8)),
                ),
                output=LayoutCase((1,)),
                value_scale=0.1,
                seed=seed + 1,
            ),
            BenchmarkCase(
                case_id="action.cl600.full.sandwich.no-reuse.batch32",
                family="action",
                operation="sandwich",
                signature=(6, 0, 0),
                inputs=(LayoutCase(tuple(range(7)), leading_shape=(32,)),) * 3,
                output=LayoutCase(tuple(range(7))),
                value_scale=0.1,
                seed=seed + 2,
            ),
            BenchmarkCase(
                case_id="action.cl600.full.sandwich.shared-pair.batch512",
                family="action",
                operation="sandwich",
                signature=(6, 0, 0),
                inputs=(
                    LayoutCase(tuple(range(7))),
                    LayoutCase(tuple(range(7)), leading_shape=(512,)),
                    LayoutCase(tuple(range(7))),
                ),
                output=LayoutCase(tuple(range(7))),
                value_scale=0.1,
                seed=seed + 3,
            ),
            BenchmarkCase(
                case_id="action.cl800.full.sandwich.no-reuse.batch16",
                family="action",
                operation="sandwich",
                signature=(8, 0, 0),
                inputs=(LayoutCase(tuple(range(9)), leading_shape=(16,)),) * 3,
                output=LayoutCase(tuple(range(9))),
                value_scale=0.1,
                seed=seed + 4,
            ),
            BenchmarkCase(
                case_id="action.cl800.full.sandwich.broadcast-reuse32x8",
                family="action",
                operation="sandwich",
                signature=(8, 0, 0),
                inputs=(
                    LayoutCase(tuple(range(9)), leading_shape=(32, 1)),
                    LayoutCase(tuple(range(9)), leading_shape=(32, 8)),
                    LayoutCase(tuple(range(9)), leading_shape=(32, 1)),
                ),
                output=LayoutCase(tuple(range(9))),
                value_scale=0.1,
                seed=seed + 5,
            ),
        )
    )
    return tuple(cases)


def exploratory_cases(families: tuple[str, ...] = ("product", "bivector_exp", "action")) -> tuple[BenchmarkCase, ...]:
    """Return deterministic semantic cases for the requested routed families."""

    unknown = set(families) - FAMILIES
    if unknown:
        raise ValueError(f"unknown benchmark families: {sorted(unknown)!r}")
    by_family = {
        "product": exploratory_product_cases,
        "bivector_exp": exploratory_bivector_exp_cases,
        "action": exploratory_action_cases,
    }
    return tuple(case for family in families for case in by_family[family]())


def full_cases(families: tuple[str, ...] = ("product", "bivector_exp", "action")) -> tuple[BenchmarkCase, ...]:
    """Return the canonical complete fixed suite.

    ``exploratory_cases`` remains the family-specific constructor; this name
    is the stable suite boundary used by the command-line runner.
    """
    return exploratory_cases(families)


AMORTIZATION_CASE_IDS = {
    "product.cl5.full-full.gp.batch1",
    "product.cl5.full-full.gp.batch1024",
    "product.cl12.vector-bivector.gp.batch64",
    "product.cl8.full-full.gp.batch16",
    "bivector_exp.cl600.even.commuting_planes.batch32",
    "action.cl400.vector.rotor.aligned",
    "action.cl400.bivector.linear.aligned",
    "action.cl600.full.sandwich.shared-pair.batch512",
    "action.cl800.full.sandwich.no-reuse.batch16",
    "action.cl800.full.sandwich.broadcast-reuse32x8",
}
