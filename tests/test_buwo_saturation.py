"""
Unit tests for a saturated surface in the build-up and washoff module.
"""

import math

import numpy as np
import pytest

from mogestpy.quality.buwo import BuildUpWashoff

BMAX = 50.0
LANDUSE_AREA = 50.0  # area * area_fraction
SATURATED_MASS = BMAX * LANDUSE_AREA


def make_model(bu_method, surface_flow, initial_buildup=SATURATED_MASS, kw=0.05):
    """Create a model for one land use with a daily time step."""
    return BuildUpWashoff(
        Bmax=BMAX,
        Nb=1.0,
        Kb=0.3,
        threshold_flow=1.0,
        Nw=1.0,
        Kw=kw,
        BuMethod=bu_method,
        WoMethod=1,
        timestep_h=24,
        initial_buildup=initial_buildup,
        area=100.0,
        area_fraction=0.5,
        surface_flow=np.array(surface_flow),
        landuse_name="urban",
    )


@pytest.mark.parametrize("bu_method", [2, 3])
def test_time_from_saturated_buildup_is_infinite(bu_method):
    model = make_model(bu_method, [0])
    time_from_buildup = {
        2: model.TimeFromBuildUpExp,
        3: model.TimeFromBuildUpSat,
    }[bu_method]

    assert time_from_buildup(BMAX) == math.inf


@pytest.mark.parametrize("bu_method", [2, 3])
def test_buildup_at_infinite_time_is_bmax(bu_method):
    model = make_model(bu_method, [0])
    buildup = {2: model.BuildUpExp, 3: model.BuildUpSat}[bu_method]

    assert buildup(math.inf) == BMAX


@pytest.mark.parametrize("bu_method", [1, 2, 3])
def test_saturated_surface_stays_at_bmax_in_dry_steps(bu_method):
    model = make_model(bu_method, [0, 0, 0])
    model.Process()

    assert model.BuildUp == pytest.approx([SATURATED_MASS] * 3)


@pytest.mark.parametrize("bu_method", [2, 3])
def test_wet_step_without_washoff_keeps_saturation(bu_method):
    model = make_model(bu_method, [5.0, 0, 0], kw=0.0)
    model.Process()

    assert model.EffectiveWashoff == [0]
    assert model.BuildUp == pytest.approx([SATURATED_MASS] * 3)


def test_long_dry_period_reaches_bmax_and_stays_after_wet_step():
    # The exponential build-up reaches Bmax in floating point after enough
    # dry days, so the next wet step without washoff starts from saturation.
    model = make_model(2, [0] * 200 + [5.0, 0], initial_buildup=10.0, kw=0.0)
    model.Process()

    assert model.BuildUp[199] == SATURATED_MASS
    assert model.BuildUp[200:] == pytest.approx([SATURATED_MASS] * 2)


def test_saturated_surface_builds_up_again_after_washoff():
    model = make_model(2, [5.0, 0])
    model.Process()

    washed = 0.05 * 5.0 * SATURATED_MASS
    left = SATURATED_MASS - washed
    start = -math.log(1 - left / LANDUSE_AREA / BMAX) / 0.3
    expected = BMAX * (1 - math.exp(-0.3 * (start + 1))) * LANDUSE_AREA

    assert model.EffectiveWashoff == pytest.approx([washed])
    assert model.BuildUp == pytest.approx([left, expected])
