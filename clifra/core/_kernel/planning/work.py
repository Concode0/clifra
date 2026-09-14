"""Pure, unweighted structural work profiles for prepared routes."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class ProductWorkProfile:
    """Minimal ordinary-product basis; the route determines its execution ledger."""

    route: str
    bulk: int
    output: int


def product_work_profile(route: str, *, interactions: int, output_width: int, full_width: int) -> ProductWorkProfile:
    if route not in {"sparse", "full_table"}:
        raise ValueError(f"unknown product route {route!r}")
    if min(interactions, output_width, full_width) < 0:
        raise ValueError("product work counts must be non-negative")
    return ProductWorkProfile(
        route,
        interactions if route == "sparse" else full_width**2,
        output_width if route == "sparse" else full_width,
    )


@dataclass(frozen=True)
class ProductCall:
    """Repeated execution of one selected child product, with its route intact."""

    profile: ProductWorkProfile
    calls: int = 1


@dataclass(frozen=True)
class BivectorExpWorkEquation:
    products: tuple[ProductCall, ...] = ()
    elementwise_cells: int = 0
    matrix_exp_order: int = 0


@dataclass(frozen=True)
class BivectorExpWorkProfile:
    """Separate closed, matrix, and Taylor execution structures."""

    route: str
    bivector_width: int
    grade4_width: int
    even_width: int
    output_width: int
    fixed_products: tuple[ProductWorkProfile, ...] = ()
    plain_products: tuple[ProductWorkProfile, ...] = ()
    scaled_products: tuple[ProductWorkProfile, ...] = ()
    square_product: ProductWorkProfile | None = None
    plain_stage_widths: tuple[int, ...] = ()
    scaled_stage_widths: tuple[int, ...] = ()

    def closed(self) -> BivectorExpWorkEquation:
        if self.route != "closed":
            raise ValueError("closed work requires the closed route")
        B, G, O = self.bivector_width, self.grade4_width, self.output_width
        return BivectorExpWorkEquation(
            tuple(ProductCall(item) for item in self.fixed_products),
            B + B * O + G * O + O,
        )

    def left_matrix_exp(self) -> BivectorExpWorkEquation:
        if self.route != "left_matrix_exp" or len(self.fixed_products) != 1:
            raise ValueError("matrix work requires one selected left-product child")
        E = self.even_width
        return BivectorExpWorkEquation((ProductCall(self.fixed_products[0], E),), self.output_width, E**3)

    def taylor_plain(self) -> BivectorExpWorkEquation:
        if self.route != "taylor":
            raise ValueError("plain Taylor work requires the Taylor route")
        return BivectorExpWorkEquation(
            tuple(ProductCall(item) for item in self.plain_products),
            sum(self.plain_stage_widths),
        )

    def taylor_scaled(self, squares: int) -> BivectorExpWorkEquation:
        if self.route != "taylor" or self.square_product is None:
            raise ValueError("scaled Taylor work requires the Taylor route and a square child")
        if isinstance(squares, bool) or not isinstance(squares, int) or not 1 <= squares <= 16:
            raise ValueError("Taylor square count must be a runtime integer in [1, 16]")
        return BivectorExpWorkEquation(
            (*tuple(ProductCall(item) for item in self.scaled_products), ProductCall(self.square_product, squares)),
            sum(self.scaled_stage_widths) + squares * self.even_width + self.output_width,
        )


@dataclass(frozen=True)
class ActionGradeWork:
    grade: int
    block_entries: int
    compound_minor_work: int
    direct_terms: int
    direct_reductions: int


@dataclass(frozen=True)
class ActionLiftEquation:
    compound_minor_work: int = 0
    direct_terms: int = 0
    direct_reductions: int = 0
    full_matrix_cells: int = 0
    dense_matvec_cells: int = 0
    block_matvec_cells: int = 0
    grade1_matvec_cells: int = 0
    output_assembly_cells: int = 0


@dataclass(frozen=True)
class ActionLiftProfile:
    """All structural grade implementations, without selecting direct grades."""

    input_width: int
    output_width: int
    grade1_entries: int
    grades: tuple[ActionGradeWork, ...]
    vector_only: bool = False

    def compound(self) -> ActionLiftEquation:
        dense = self.input_width * self.output_width
        return ActionLiftEquation(
            compound_minor_work=sum(item.compound_minor_work for item in self.grades),
            full_matrix_cells=0 if self.vector_only else dense,
            dense_matvec_cells=dense,
        )

    def hybrid(self, direct_grades: tuple[int, ...]) -> ActionLiftEquation:
        selected = set(direct_grades)
        available = {item.grade for item in self.grades}
        if not selected or len(selected) != len(direct_grades) or not selected <= available:
            raise ValueError("hybrid action needs unique available direct grades")
        direct = tuple(item for item in self.grades if item.grade in selected)
        compound = tuple(item for item in self.grades if item.grade not in selected)
        return ActionLiftEquation(
            compound_minor_work=sum(item.compound_minor_work for item in compound),
            direct_terms=sum(item.direct_terms for item in direct),
            direct_reductions=sum(item.direct_reductions for item in direct),
            block_matvec_cells=sum(item.block_entries for item in compound),
            grade1_matvec_cells=self.grade1_entries,
            output_assembly_cells=self.output_width,
        )

    def direct(self) -> ActionLiftEquation:
        return self.hybrid(tuple(item.grade for item in self.grades))


def action_lift_profile(n: int, input_grades: tuple[int, ...], output_grades: tuple[int, ...]) -> ActionLiftProfile:
    shared = set(input_grades) & set(output_grades)
    grades = []
    for g in sorted(shared):
        if g < 2:
            continue
        block = comb(n, g) ** 2
        minor = block * (2 if g == 2 else 9 if g == 3 else g**3)
        terms = sum(comb(n, j + 1) * comb(n, g - j - 1) * (n - g + j + 1) for j in range(g))
        reductions = sum(comb(n, j + 1) * comb(n, g - j - 1) * (n - g + j) for j in range(g))
        grades.append(ActionGradeWork(g, block, minor, terms, reductions))
    return ActionLiftProfile(
        sum(comb(n, g) for g in input_grades),
        sum(comb(n, g) for g in output_grades),
        n**2 if 1 in shared else 0,
        tuple(grades),
        input_grades == output_grades == (1,),
    )


@dataclass(frozen=True)
class ActionWorkProfile:
    """Route components; `lift` retains all possible equations, not a minimum."""

    route: str
    grade: int
    generator_terms: int = 0
    reflection_cells: int = 0
    vector_matrix_exp_order: int = 0
    lift: ActionLiftProfile | None = None
    product_children: tuple[ProductWorkProfile, ...] = ()
    exponential_child: BivectorExpWorkProfile | None = None
    full_action_matrix_order: int = 0
    full_action_cells: int = 0
