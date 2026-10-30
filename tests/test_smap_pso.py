"""
Tests for the PSO calibration of the SMAP models, with a stand-in for pyswarms.
"""

import sys
import types

import numpy as np
import pytest

from mogestpy.quantity.hydrological.smap import SmapD, SmapM


class FakeGlobalBestPSO:
    """Return the middle of the bounds without running a swarm."""

    def __init__(self, n_particles, dimensions, options, bounds):
        self.lower, self.upper = (np.array(b) for b in bounds)

    def optimize(self, objective, iters):
        position = (self.lower + self.upper) / 2
        return objective(position), position


@pytest.fixture
def fake_pyswarms(monkeypatch):
    module = types.ModuleType("pyswarms")
    module.single = types.SimpleNamespace(GlobalBestPSO=FakeGlobalBestPSO)
    monkeypatch.setitem(sys.modules, "pyswarms", module)


def run_pso(model, **kwargs):
    prec = [10.0, 0.0, 5.0, 20.0, 0.0, 0.0]
    etp = [3.0] * len(prec)
    obs = [1.0, 0.8, 0.9, 2.0, 1.2, 1.0]
    return model.pso_calibrate(prec, etp, obs, ["Str"], **kwargs)


@pytest.mark.parametrize("smap_class", [SmapD, SmapM])
def test_pso_calibrate_is_quiet_by_default(smap_class, fake_pyswarms, capsys):
    run_pso(smap_class())

    assert "bounds" not in capsys.readouterr().out


@pytest.mark.parametrize("smap_class", [SmapD, SmapM])
def test_pso_calibrate_prints_bounds_when_verbose(smap_class, fake_pyswarms, capsys):
    model = smap_class()
    lower, upper = model.bounds()["Str"]
    run_pso(model, verbose=True)

    out = capsys.readouterr().out
    assert f"Lower bounds: ({lower},), Upper bounds: ({upper},)" in out


@pytest.mark.parametrize("smap_class", [SmapD, SmapM])
def test_pso_calibrate_sets_best_parameters(smap_class, fake_pyswarms):
    model = smap_class()
    lower, upper = model.bounds()["Str"]
    result = run_pso(model)

    assert result["best_params"]["Str"] == pytest.approx((lower + upper) / 2)
    assert model.Str == pytest.approx((lower + upper) / 2)
