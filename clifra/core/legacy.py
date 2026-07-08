# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Central compatibility names for legacy public operation aliases.

The aliases here ship as quiet compatibility in 1.1.1 and are planned for
removal in the next release. Keep this module as the single compatibility fence
so the future removal is mechanical.
"""

from __future__ import annotations

LEGACY_COMPAT_VERSION = "1.1.1"
LEGACY_REMOVAL_TARGET = "next release after 1.1.1"

LEGACY_OPERATION_ALIASES = {
    "inner": "symmetric_product",
    "inner_product": "symmetric_product",
    "commutator": "commutator_product",
    "anti_commutator": "anti_commutator_product",
    "anticommutator": "anti_commutator_product",
    "dual": "pseudoscalar_product",
    "norm_sq": "signature_norm_squared",
    "exp": "bivector_exp",
}

LEGACY_PRODUCT_OP_ALIASES = {
    "inner": "symmetric_product",
    "inner_product": "symmetric_product",
    "commutator": "commutator_product",
    "anti_commutator": "anti_commutator_product",
    "anticommutator": "anti_commutator_product",
}

LEGACY_SYMBOL_ALIASES = {
    "AntiCommutatorLayer": "AntiCommutatorProductLayer",
    "CommutatorLayer": "CommutatorProductLayer",
    "DualExecutor": "PseudoscalarProductExecutor",
    "DualPlan": "PseudoscalarProductPlan",
    "InnerProductLayer": "SymmetricProductLayer",
    "NormSquaredExecutor": "SignatureNormSquaredExecutor",
    "NormSquaredPlan": "SignatureNormSquaredPlan",
    "build_dual_plan": "build_pseudoscalar_product_plan",
    "build_norm_squared_plan": "build_signature_norm_squared_plan",
}

PREFERRED_PRODUCT_OPS = {
    "gp": ("gp", "geometric_product"),
    "geometric_product": ("gp", "geometric_product"),
    "wedge": ("wedge", "wedge"),
    "outer": ("wedge", "wedge"),
    "symmetric_product": ("inner", "symmetric_product"),
    "commutator_product": ("commutator", "commutator_product"),
    "anti_commutator_product": ("anti_commutator", "anti_commutator_product"),
    "left_contraction": ("left_contraction", "left_contraction"),
    "right_contraction": ("right_contraction", "right_contraction"),
}


def canonical_operation_name(name: str) -> str:
    """Return the preferred operation name for public compatibility aliases."""
    normalized = str(name).lower()
    preferred = LEGACY_OPERATION_ALIASES.get(normalized)
    if preferred is not None:
        return preferred
    return normalized


def canonical_product_alias(op: str) -> str:
    """Return the preferred product operation name for legacy product aliases."""
    normalized = str(op).lower()
    preferred = LEGACY_PRODUCT_OP_ALIASES.get(normalized)
    if preferred is not None:
        return preferred
    return normalized


def product_method_entry(op: str) -> tuple[str, str]:
    """Return ``(planner_op, algebra_method)`` for a product operation alias."""
    normalized = canonical_product_alias(op)
    try:
        return PREFERRED_PRODUCT_OPS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(PREFERRED_PRODUCT_OPS | LEGACY_PRODUCT_OP_ALIASES))
        raise ValueError(f"Unsupported product op {op!r}. Supported ops: {supported}") from exc


__all__ = [
    "LEGACY_OPERATION_ALIASES",
    "LEGACY_PRODUCT_OP_ALIASES",
    "LEGACY_SYMBOL_ALIASES",
    "LEGACY_COMPAT_VERSION",
    "LEGACY_REMOVAL_TARGET",
    "PREFERRED_PRODUCT_OPS",
    "canonical_operation_name",
    "canonical_product_alias",
    "product_method_entry",
]
