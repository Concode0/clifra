"""Layer-facing storage facade.

The storage contracts live in :mod:`clifra.core.storage`; this module keeps the
runtime package from owning a second copy of the same layer metadata logic.
"""

from clifra.core.storage import LayerStorage, resolve_layer_layout, resolve_layer_storage

__all__ = ["LayerStorage", "resolve_layer_layout", "resolve_layer_storage"]
