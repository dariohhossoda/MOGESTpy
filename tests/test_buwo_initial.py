"""
Unit tests for the initial build-up of the build-up and washoff module.
"""

import math

import numpy as np
import pytest

from mogestpy.quality.buwo import BuildUpWashoff

BMAX = 50.0
KB = 0.3
KW = 0.05
FLOW = 5.0
INITIAL_BUILDUP = 10.0
LANDUSE_AREA = 50.0  # area * area_fraction


def run_model(surface_flow):
    """Run the exponential build-up and washoff with a daily time step."""
    model = BuildUpWashoff(
        Bmax=BMAX,
        Nb=1.0,
        Kb=KB,
        threshold_flow=1.0,
        Nw=1.0,
        Kw=KW,
        BuMethod=2,
        WoMethod=1,
        timestep_h=24,
        initial_buildup=INITIAL_BUILDUP,
        area=100.0,
        area_fraction=0.5,
        surface_flow=np.array(surface_flow),
        landuse_name="urban",
    )
    model.Process()
    return model


def exponential_buildup_after(days, mass):
    """Mass on the surface after some dry days, starting from a given mass."""
    specific = mass / LANDUSE_AREA
    start = -math.log(1 - specific / BMAX) / KB
    return BMAX * (1 - math.exp(-KB * (start + days))) * LANDUSE_AREA


def test_wet_first_step_washes_initial_mass():
    model = run_model([FLOW, 0])

    assert model.Washoff[0] == pytest.approx(KW * FLOW * INITIAL_BUILDUP)
    assert model.BuildUp[0] == pytest.approx(INITIAL_BUILDUP * (1 - KW * FLOW))


def test_buildup_continues_from_mass_left_after_washoff():
    model = run_model([FLOW, 0])

    assert model.BuildUp[1] == pytest.approx(
        exponential_buildup_after(1, model.BuildUp[0])
    )


def test_dry_first_step_starts_from_initial_mass():
    model = run_model([0, 0])

    assert model.BuildUp == pytest.approx(
        [exponential_buildup_after(day, INITIAL_BUILDUP) for day in (1, 2)]
    )
