"""
Unit tests for the build-up and washoff module using pytest.
"""

import numpy as np
import pytest

from mogestpy.quality.buwo import BuildUpWashoff


def make_model(bu_method, wo_method=1, surface_flow=None):
    """Create a model for one land use with a daily time step."""
    if surface_flow is None:
        surface_flow = np.array([0, 0, 0, 5.0, 0, 0])
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
        initial_buildup=10.0,
        area=100.0,
        area_fraction=0.5,
        surface_flow=surface_flow,
        landuse_name="urban",
    )


def test_buildup_pow_below_bmax():
    model = make_model(bu_method=1)
    assert model.BuildUpPow(2.0) == pytest.approx(0.6)


def test_buildup_pow_limited_to_bmax():
    model = make_model(bu_method=1)
    assert model.BuildUpPow(1000.0) == 50.0


def test_process_with_power_buildup():
    model = make_model(bu_method=1, surface_flow=np.array([0, 0, 0]))
    model.Process()

    # Initial build-up of 10 over 50 units of area: 0.2 per unit area, which
    # the power equation reaches after 2/3 day. Each dry day adds one day.
    expected = [0.3 * (2 / 3 + day) * 50 for day in (1, 2, 3)]
    assert model.BuildUp == pytest.approx(expected)


@pytest.mark.parametrize("bu_method", [1, 2, 3])
@pytest.mark.parametrize("wo_method", [1, 2, 3])
def test_process_runs_for_every_method(bu_method, wo_method):
    model = make_model(bu_method, wo_method)
    model.Process()

    assert len(model.BuildUp) == len(model.SurfaceFlow)
    assert len(model.EffectiveWashoff) == 1
    assert all(b >= 0 for b in model.BuildUp)
