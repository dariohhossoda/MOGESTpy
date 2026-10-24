"""
Unit tests for the washoff equations of the build-up and washoff module.
"""

import numpy as np
import pytest

from mogestpy.quality.buwo import BuildUpWashoff

KW = 0.05
NW = 1.0
FLOW = 5.0
LANDUSE_AREA = 50.0  # area * area_fraction


def run_model(wo_method, bmax=50.0):
    """Run one dry day followed by one wet day with a daily time step."""
    model = BuildUpWashoff(
        Bmax=bmax,
        Nb=1.0,
        Kb=0.3,
        threshold_flow=1.0,
        Nw=NW,
        Kw=KW,
        BuMethod=2,
        WoMethod=wo_method,
        timestep_h=24,
        initial_buildup=10.0,
        area=100.0,
        area_fraction=0.5,
        surface_flow=np.array([0, FLOW]),
        landuse_name="urban",
    )
    model.Process()
    return model


def test_exponential_washoff_uses_buildup():
    model = run_model(wo_method=1)
    buildup = model.BuildUp[0]

    assert model.Washoff[-1] == pytest.approx(KW * FLOW**NW * buildup)


def test_rating_curve_washoff_uses_landuse_area():
    model = run_model(wo_method=2)

    assert model.Washoff[-1] == pytest.approx(KW * (1000 * FLOW * LANDUSE_AREA) ** NW)


def test_emc_washoff_uses_landuse_area():
    model = run_model(wo_method=3)

    assert model.Washoff[-1] == pytest.approx(KW * FLOW * LANDUSE_AREA)


@pytest.mark.parametrize("wo_method", [2, 3])
def test_flow_washoff_does_not_depend_on_buildup(wo_method):
    low = run_model(wo_method, bmax=20.0)
    high = run_model(wo_method, bmax=80.0)

    assert low.Washoff[-1] == pytest.approx(high.Washoff[-1])


@pytest.mark.parametrize("wo_method", [1, 2, 3])
def test_effective_washoff_limited_to_buildup(wo_method):
    model = run_model(wo_method)

    assert model.EffectiveWashoff[-1] <= model.BuildUp[0]
    assert model.BuildUp[-1] == pytest.approx(
        model.BuildUp[0] - model.EffectiveWashoff[-1]
    )
