import pytest

from clifra.core.runtime.algebra import AlgebraContext

DEVICE = "cpu"


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

# -- Module-scoped (used by test_geodesic.py - exact name match) ----------
@pytest.fixture(scope="module")
def alg2():
    return AlgebraContext(p=2, q=0, device=DEVICE)


@pytest.fixture(scope="module")
def alg3():
    return AlgebraContext(p=3, q=0, device=DEVICE)


@pytest.fixture(scope="module")
def alg31():
    return AlgebraContext(p=3, q=1, device=DEVICE)
