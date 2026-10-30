"""
Tests for the SMAP calibration, parameter bounds and deprecated method names.
"""

import pytest

from mogestpy.quantity.hydrological.smap import SmapD, SmapM

DAILY_PREC = [0.0, 0.0, 12.5, 30.1, 4.2, 0.0, 0.0, 18.7, 2.3, 0.0] * 6
DAILY_ETP = [4.5] * len(DAILY_PREC)

MONTHLY_PREC = [250, 210, 180, 80, 60, 40, 30, 35, 70, 120, 160, 230] * 3
MONTHLY_ETP = [120, 110, 105, 85, 70, 60, 65, 80, 95, 110, 115, 120] * 3


def make_smapm(**kwargs):
    params = dict(Ad=17800.0, Crec=0.2, Tuin=0.8, Ebin=204.25, Pes=4.0, kkt=1.5)
    params.update(kwargs)
    return SmapM(**params)


@pytest.mark.parametrize("method", ["calibrate", "fast_calibrate"])
def test_smapm_calibration_recovers_parameter(method):
    observed = make_smapm(Str=1200.0).run_to_list(MONTHLY_PREC, MONTHLY_ETP)
    smap = make_smapm(Str=500.0)

    result = getattr(smap, method)(
        MONTHLY_PREC, MONTHLY_ETP, observed, variables=["Str"]
    )

    assert result.x[0] == pytest.approx(1200.0, rel=0.05)
    assert result.fun < -0.99


def test_smapm_fast_calibrate_sets_rounded_parameter():
    observed = make_smapm(Str=1200.0).run_to_list(MONTHLY_PREC, MONTHLY_ETP)
    smap = make_smapm(Str=500.0)

    result = smap.fast_calibrate(MONTHLY_PREC, MONTHLY_ETP, observed, ["Str"])

    assert result.x[0] == round(result.x[0], 4)
    assert smap.Str == result.x[0]


def test_calibrate_with_custom_objective():
    observed = SmapD(Str=1200.0).run_to_list(DAILY_PREC, DAILY_ETP)

    def sse(obs, sim):
        return sum((o - s) ** 2 for o, s in zip(obs, sim))

    result = SmapD().calibrate(
        DAILY_PREC, DAILY_ETP, observed, variables=["Str"], obj_func=sse
    )

    assert result.x[0] == pytest.approx(1200.0, rel=0.05)
    assert result.fun < 1e-3


@pytest.mark.parametrize("smap_class", [SmapD, SmapM])
@pytest.mark.parametrize("method", ["calibrate", "fast_calibrate"])
def test_calibration_rejects_unknown_variable(smap_class, method):
    with pytest.raises(ValueError, match="unknown"):
        getattr(smap_class(), method)([1.0], [1.0], [1.0], variables=["unknown"])


@pytest.mark.parametrize("smap_class", [SmapD, SmapM])
def test_check_bounds(smap_class):
    smap = smap_class()
    lower, upper = smap.bounds()["Str"]

    assert smap.check_bounds({"Str": lower, "Crec": smap.bounds()["Crec"][1]})
    assert not smap.check_bounds({"Str": upper + 1})
    assert not smap.check_bounds({"Str": lower - 1})
    # Names without bounds are ignored
    assert smap.check_bounds({"not_a_parameter": -1})


@pytest.mark.parametrize("smap_class", [SmapD, SmapM])
def test_str_lists_parameters(smap_class):
    text = str(smap_class(Str=1234.5))

    assert "Str = 1234.5" in text
    assert "Crec" in text


def test_smapd_warns_when_tuin_looks_like_a_percentage():
    with pytest.warns(UserWarning, match="Tuin"):
        SmapD(Tuin=80)


@pytest.mark.parametrize(
    "old_name, new_name, args",
    [
        ("RunStep", "run_step", (12.5, 4.5)),
        ("Run", "run", (DAILY_PREC, DAILY_ETP)),
        ("RunToList", "run_to_list", (DAILY_PREC, DAILY_ETP)),
    ],
)
def test_smapd_deprecated_run_names(old_name, new_name, args):
    with pytest.warns(DeprecationWarning, match=f"{old_name} is deprecated"):
        old = getattr(SmapD(), old_name)(*args)
    new = getattr(SmapD(), new_name)(*args)

    if old_name == "Run":
        old, new = list(old), list(new)
    assert old == pytest.approx(new)


def test_smapd_deprecated_calibrate_name():
    prec, etp = DAILY_PREC[:20], DAILY_ETP[:20]
    observed = SmapD(Str=1200.0).run_to_list(prec, etp)

    with pytest.warns(DeprecationWarning, match="Calibrate is deprecated"):
        result = SmapD().Calibrate(prec, etp, observed, variables=["Str"])

    assert result.fun < -0.9
