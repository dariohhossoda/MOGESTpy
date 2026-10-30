"""
Unit tests for the argument checks of the build-up and washoff module.
"""

import numpy as np
import pytest

from mogestpy.quality.buwo import BuildUpWashoff

MAX_BUILDUP = 50.0 * 100.0 * 0.5  # Bmax * area * area_fraction


def make_model(bu_method=2, wo_method=1, initial_buildup=10.0):
    """Create a model for one land use with a daily time step."""
    return BuildUpWashoff(
        Bmax=50.0,
        Nb=1.0,
        Kb=0.3,
        threshold_flow=1.0,
        Nw=1.2,
        Kw=0.05,
        BuMethod=bu_method,
        WoMethod=wo_method,
        timestep_h=24,
        initial_buildup=initial_buildup,
        area=100.0,
        area_fraction=0.5,
        surface_flow=np.array([0, 0, 5.0]),
        landuse_name="urban",
    )


@pytest.mark.parametrize("bu_method", [0, 4, "2", None])
def test_invalid_buildup_method_raises(bu_method):
    with pytest.raises(ValueError, match="BuMethod"):
        make_model(bu_method=bu_method)


@pytest.mark.parametrize("wo_method", [0, 4, "1", None])
def test_invalid_washoff_method_raises(wo_method):
    with pytest.raises(ValueError, match="WoMethod"):
        make_model(wo_method=wo_method)


@pytest.mark.parametrize("initial_buildup", [-1.0, MAX_BUILDUP + 1.0])
def test_initial_buildup_out_of_range_raises(initial_buildup):
    with pytest.raises(ValueError, match="initial_buildup"):
        make_model(initial_buildup=initial_buildup)


@pytest.mark.parametrize("initial_buildup", [0.0, 10.0, MAX_BUILDUP])
@pytest.mark.parametrize("bu_method", [1, 2, 3])
def test_initial_buildup_within_range_runs(bu_method, initial_buildup):
    model = make_model(bu_method=bu_method, initial_buildup=initial_buildup)
    model.Process()

    assert all(0 <= b <= MAX_BUILDUP for b in model.BuildUp)
