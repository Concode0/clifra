# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
from hypothesis import HealthCheck, settings

from clifra.core.algebra import AlgebraContext

DEVICE = "cpu"

_PBT_SETTINGS = {
    "deadline": None,
    "print_blob": True,
    "suppress_health_check": (HealthCheck.function_scoped_fixture,),
}

settings.register_profile("standard", settings(), max_examples=100, **_PBT_SETTINGS)
settings.register_profile("full", settings.get_profile("standard"), max_examples=400)
settings.load_profile("standard")


@pytest.fixture(autouse=True)
def isolated_compilation_cache(request):
    """Each compilation test measures its own plans, not prior tests' specializations."""
    if "compile" not in request.node.name and "fullgraph" not in request.node.name:
        yield
        return
    import torch

    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# -- Function-scoped (default) ------------------------------------------
@pytest.fixture
def algebra_2d():
    return AlgebraContext(p=2, q=0, device=DEVICE)


@pytest.fixture
def algebra_3d():
    return AlgebraContext(p=3, q=0, device=DEVICE)


@pytest.fixture
def algebra_4d():
    return AlgebraContext(p=4, q=0, device=DEVICE)


@pytest.fixture
def algebra_spacetime():
    return AlgebraContext(p=1, q=3, device=DEVICE)


@pytest.fixture
def algebra_minkowski():
    return AlgebraContext(p=2, q=1, device=DEVICE)


@pytest.fixture
def algebra_conformal():
    return AlgebraContext(p=4, q=1, device=DEVICE)
