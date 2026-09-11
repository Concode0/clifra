"""Versioned mathematical workloads; bounded, family-owned expansion tracks."""

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    operation: str
    signature: tuple[int, int, int]
    inputs: tuple[tuple[int, ...], ...]
    output: tuple[int, ...]
    leading: tuple[tuple[int, ...], ...]
    reason: str
    structure: str = "generic"
    magnitude: float = 0.5
    pairwise: bool = False
    revision: int = 1


def core_cases():
    cases = []

    def add(id, family, op, sig, inputs, output, leading, reason, **kw):
        cases.append(Case(id, family, op, sig, inputs, output, leading, reason, **kw))

    for name, sig, batch, op in (
        ("gp/e3/full/single", (3, 0, 0), (), "geometric_product"),
        ("gp/e6/full/b32", (6, 0, 0), (32,), "geometric_product"),
        ("wedge/e6/full/b32", (6, 0, 0), (32,), "wedge"),
        ("gp/pga3/full/b32", (3, 0, 1), (32,), "geometric_product"),
    ):
        full = tuple(range(sum(sig) + 1))
        add(
            "product/" + name,
            "product",
            op,
            sig,
            (full, full),
            full,
            (batch, batch),
            "Full-layout dense/structurally sparse route competition",
        )
    for name, n, left, right, out, batch, op in (
        ("gp/e16/vector/b64", 16, (1,), (1,), (0, 2), 64, "geometric_product"),
        ("scalar/e32/vector/b256", 32, (1,), (1,), (0,), 256, "geometric_product"),
        ("commutator/e8/bivector-vector/b64", 8, (2,), (1,), (1,), 64, "commutator_product"),
        ("gp/e6/mixed-full/b32", 6, (0, 2), tuple(range(7)), tuple(range(7)), 32, "geometric_product"),
        ("gp/e8/scalar-vector/b256", 8, (0,), (1,), (1,), 256, "geometric_product"),
    ):
        add(
            "product/" + name,
            "product",
            op,
            (n, 0, 0),
            (left, right),
            out,
            ((batch,), (batch,)),
            "Compact support and specialized product execution",
        )
    add(
        "product/gp/e5/vector/pairwise",
        "product",
        "geometric_product",
        (5, 0, 0),
        ((1,), (1,)),
        (0, 2),
        ((2, 16), (1, 24)),
        "Pairwise contraction with broadcast gradients",
        pairwise=True,
    )
    for name, sig, structure, magnitude, batch, out in (
        ("e3/simple/l1-0p25/b32", (3, 0, 0), "simple", 0.25, 32, None),
        ("e4/two-plane/l1-0p5/b32", (4, 0, 0), "planes", 0.5, 32, None),
        ("cl3-1-0/generic/l1-0p5/b16", (3, 1, 0), "generic", 0.5, 16, None),
        ("pga3/null/l1-0p5/b32", (3, 0, 1), "null", 0.5, 32, None),
        ("e6/generic/l1-0p5/b16", (6, 0, 0), "generic", 0.5, 16, None),
        ("e6/generic/l1-4/b16", (6, 0, 0), "generic", 4.0, 16, None),
        ("e6/generic/l1-0p5/scalar/b16", (6, 0, 0), "generic", 0.5, 16, (0,)),
    ):
        add(
            "exp/" + name,
            "bivector_exp",
            "bivector_exp",
            sig,
            ((2,),),
            tuple(range(0, sum(sig) + 1, 2)) if out is None else out,
            ((batch,),),
            "Closed/general, scaling, null or output-pruned exponential regime",
            structure=structure,
            magnitude=magnitude,
        )
    for name, sig, grades, leading, structure in (
        ("e3/vector/shared-bivector", (3, 0, 0), (1,), ((128,), ()), "generic"),
        ("e4/full/shared-bivector", (4, 0, 0), tuple(range(5)), ((32,), ()), "generic"),
        ("e6/grade2/broadcast-bivector", (6, 0, 0), (2,), ((8, 1), (4,)), "generic"),
        ("pga3/vector/null-bivector", (3, 0, 1), (1,), ((64,), ()), "null"),
    ):
        add(
            "action/" + name,
            "action",
            "versor",
            sig,
            (grades, (2,)),
            grades,
            leading,
            "Induced action representation and parameter broadcasting",
            structure=structure,
        )
    for id, family, op, sig, inputs, out in (
        ("unary/e8/reverse/full/b256", "unary", "reverse", (8, 0, 0), (tuple(range(9)),), tuple(range(9))),
        ("grade/e8/project/012-to-1/b256", "unary", "grade_projection", (8, 0, 0), ((0, 1, 2),), (1,)),
        ("form/cl3-1-1/signature-norm/12/b256", "metric", "signature_norm_squared", (3, 1, 1), ((1, 2),), (0,)),
        ("permutation/pga3/pseudoscalar/12/b256", "permutation", "pseudoscalar_product", (3, 0, 1), ((1, 2),), (2, 3)),
        ("form/e8/lane-grade-energy/012/b256", "forms", "lane_grade_energy", (8, 0, 0), ((0, 1, 2),), ()),
        (
            "form/cl3-1-0/conjugate-scalar/aligned/b256",
            "forms",
            "conjugate_scalar_form",
            (3, 1, 0),
            ((0, 1), (1, 2)),
            (0,),
        ),
        ("geometry/e3/reflect/full/b64", "geometry", "reflect", (3, 0, 0), (tuple(range(4)), (1,)), tuple(range(4))),
    ):
        batch = 64 if family == "geometry" else 256
        add(
            id,
            family,
            op,
            sig,
            inputs,
            out,
            tuple((batch,) for _ in inputs),
            "Unary, form, storage or geometric composition regression coverage",
        )
    return tuple(cases)


SMOKE = (
    "product/gp/e3/full/single",
    "exp/e3/simple/l1-0p25/b32",
    "action/e3/vector/shared-bivector",
    "grade/e8/project/012-to-1/b256",
    "form/cl3-1-1/signature-norm/12/b256",
    "permutation/pga3/pseudoscalar/12/b256",
    "form/e8/lane-grade-energy/012/b256",
)
TRACKS = (
    "product-full",
    "product-grades",
    "product-shapes",
    "exp-domains",
    "exp-scaling",
    "action-grades",
    "operations",
    "product-batches",
    "product-compact",
    "exp-batches",
    "action-batches",
    "action-reflections",
    "forms",
)
PRODUCT_OPS = (
    "geometric_product",
    "wedge",
    "left_contraction",
    "right_contraction",
    "symmetric_product",
    "commutator_product",
    "anti_commutator_product",
)


def sweep(track, dimensions):
    """Each track varies its own meaningful axes, with fixed orthogonal settings."""
    if track not in TRACKS:
        raise ValueError(f"unknown track: {track}")
    seen = set()
    for n in dimensions:
        if not 2 <= n <= 63:
            raise ValueError("sweep dimensions must lie in [2, 63]")
        if track.startswith("exp-") and n > 12:
            raise ValueError("materialized exponential tracks stop at dimension 12")
        if track in {"product-full", "product-batches"} and n > 12:
            raise ValueError("full-layout track stops at dimension 12; normal resource limits still apply")
        full, even = tuple(range(n + 1)), tuple(range(0, n + 1, 2))
        specs = ((n, 0, 0), (n - 1, 1, 0), (n - 1, 0, 1))
        declarations = []
        if track == "product-full":
            declarations = [
                ("product", op, s, (full, full), full, ((16,), (16,)), "generic", 0.5, False)
                for s in specs
                for op in PRODUCT_OPS
            ]
        elif track == "product-batches":
            declarations = [
                ("product", op, s, (full, full), full, ((batch,), (batch,)), "generic", 0.5, False)
                for s in (specs[0], specs[2])
                for op in ("geometric_product", "wedge", "left_contraction", "commutator_product")
                for batch in (1, 8, 64, 512)
            ]
        elif track == "product-compact":
            regimes = [
                ("geometric_product", (0,), (1,), (1,)),
                ("geometric_product", (1,), (1,), (0,)),
                ("geometric_product", (1,), (1,), (0, 2)),
                ("wedge", (1,), (1,), (2,)),
                ("left_contraction", (2,), (1,), (1,)),
                ("right_contraction", (2,), (1,), (1,)),
                ("commutator_product", (2,), (1,), (1,)),
                ("geometric_product", (2,), (2,), tuple(g for g in (0, 2, 4) if g <= n)),
            ]
            if n <= 8:
                regimes += [("geometric_product", (0, 2), full, full)]
            declarations = [
                ("product", op, specs[0], (a, b), out, ((batch,), (batch,)), "generic", 0.5, False)
                for op, a, b, out in regimes
                for batch in (1, 64, 512)
            ]
        elif track == "product-grades":
            # All homogeneous grade triples, including zero interaction projections.
            declarations = (
                ("product", op, (n, 0, 0), ((a,), (b,)), (c,), ((1,), (1,)), "generic", 0.5, False)
                for op in PRODUCT_OPS
                for a, b, c in product(range(n + 1), repeat=3)
            )
        elif track == "product-shapes":
            declarations = [
                ("product", "geometric_product", (n, 0, 0), ((1,), (1,)), (0, 2), shapes, "generic", 0.5, pairwise)
                for shapes, pairwise in [
                    (((), ()), False),
                    (((256,), ()), False),
                    (((8, 1), (4,)), False),
                    (((2, 16), (1, 24)), True),
                ]
            ]
        elif track == "exp-batches":
            declarations = [
                ("bivector_exp", "bivector_exp", s, ((2,),), out, ((batch,),), "generic", scale, False)
                for s in (specs[0], specs[1])
                for out in (even, (0,))
                for batch in (2, 16, 128)
                for scale in (0.5, 4.0)
            ]
        elif track.startswith("exp-"):
            regimes = [(s, kind, 0.5) for s in specs for kind in ("simple", "planes", "generic")]
            regimes += [(specs[-1], "null", 0.5)]
            if track == "exp-scaling":
                regimes = [
                    ((n, 0, 0), kind, scale)
                    for kind in ("generic", "mixed")
                    for scale in (0.0, 0.99, 1.0, 1.01, 4.0, 16.0)
                ]
            declarations = [
                ("bivector_exp", "bivector_exp", s, ((2,),), out, ((4,),), kind, scale, False)
                for s, kind, scale in regimes
                for out in (even, (0,), (0, 2))
            ]
        elif track in {"action-batches", "action-reflections"}:
            parameter = (1,) if track == "action-reflections" else (2,)
            declarations = [
                ("action", "versor", specs[0], (g, parameter), g, shapes, "generic", 0.5, False)
                for g in dict.fromkeys(((1,), (2,), full))
                for shapes in (((2,), (2,)), ((16,), (16,)), ((128,), (128,)), ((128, 1), (2,)))
            ]
        elif track == "action-grades":
            declarations = [
                ("action", "versor", s, (g, (2,)), g, ((8, 1), (4,)), "generic", 0.5, False)
                for s in specs
                for g in dict.fromkeys(((1,), (2,), (min(3, n),), (min(4, n),), full))
            ]
        elif track == "forms":
            declarations = [
                (family, op, s, inputs, out, tuple((batch,) for _ in inputs), "generic", 0.5, False)
                for s in specs
                for family, op, inputs, out in (
                    ("metric", "signature_norm_squared", ((0, 1, 2),), (0,)),
                    ("forms", "lane_grade_energy", ((0, 1, 2),), ()),
                    ("forms", "conjugate_scalar_form", ((0, 1, 2), (0, 1, 2)), (0,)),
                    ("forms", "conjugate_scalar_form", ((0, 1), (1, 2)), (0,)),
                    ("unary", "reverse", ((2,),), (2,)),
                    ("permutation", "pseudoscalar_product", ((1, 2),), (n - 2, n - 1)),
                )
                for batch in (16, 1024)
            ]
        else:
            declarations = [
                ("unary", op, (n, 0, 0), ((0, 1, 2),), out, ((64,),), "generic", 0.5, False)
                for op in ("identity", "reverse", "grade_involution", "clifford_conjugation", "grade_projection")
                for out in ((0, 1, 2), (1,), (n,))
            ]
        for family, op, sig, inputs, out, leading, structure, magnitude, pairwise in declarations:
            grades = "-".join("g" + ".".join(map(str, g)) for g in inputs)
            shapes = "-".join("x".join(map(str, s)) or "scalar" for s in leading)
            id = (
                f"sweep/{track}/{op}/cl{'-'.join(map(str, sig))}/{grades}-to-g{'.'.join(map(str, out))}"
                f"/{shapes}/{structure}-l1-{magnitude:g}/{'pairwise' if pairwise else 'broadcast'}"
            )
            if id in seen:
                continue
            seen.add(id)
            yield Case(
                id, family, op, sig, inputs, out, leading, f"Bounded {track} track", structure, magnitude, pairwise
            )
