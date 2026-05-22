"""Layer-facing layout-contract facade.

The layout contracts live in :mod:`clifra.core.storage`; this module keeps the
runtime package from owning a second copy of the same layer metadata logic.
"""

from clifra.core.storage import LayerLayout, resolve_layer_layout, resolve_layer_layout_contract

__all__ = ["LayerLayout", "resolve_layer_layout", "resolve_layer_layout_contract"]
