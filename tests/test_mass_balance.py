"""
Unit tests for the reservoir mass balance module using pytest.
"""

import numpy as np
import pytest

from mogestpy.quantity.hydrological.mass_balance import MassBalance


def test_constant_net_inflow_gives_linear_volume():
    model = MassBalance(Qin=[10.0] * 4, Qout=[4.0] * 4, Delta_t=2.0, Vin=100.0)

    assert model.Volume_out() == pytest.approx([100.0, 112.0, 124.0, 136.0])


def test_zero_net_flow_keeps_volume():
    model = MassBalance(
        Qin=[5.0, 7.0, 3.0], Qout=[5.0, 7.0, 3.0], Delta_t=1.0, Vin=50.0
    )

    assert model.Volume_out() == pytest.approx([50.0, 50.0, 50.0])


def test_uses_rates_at_the_end_of_each_interval():
    # V[i+1] = V[i] + (Qin[i+1] - Qout[i+1]) * Delta_t: the first rates are
    # not used, only the ones at the end of each interval.
    model = MassBalance(
        Qin=[1000.0, 3.0, 6.0], Qout=[0.0, 1.0, 2.0], Delta_t=10.0, Vin=0.0
    )

    assert model.Volume_out() == pytest.approx([0.0, 20.0, 60.0])


def test_net_outflow_reduces_volume():
    model = MassBalance(Qin=[0.0, 0.0], Qout=[0.0, 3.0], Delta_t=5.0, Vin=20.0)

    assert model.Volume_out() == pytest.approx([20.0, 5.0])


def test_single_time_step_returns_initial_volume():
    model = MassBalance(Qin=[2.0], Qout=[1.0], Delta_t=1.0, Vin=7.5)

    assert model.Volume_out() == [7.5]


def test_accepts_numpy_arrays():
    lists = MassBalance(Qin=[1.0, 2.0, 3.0], Qout=[0.5] * 3, Delta_t=3.0, Vin=1.0)
    arrays = MassBalance(
        Qin=np.array([1.0, 2.0, 3.0]), Qout=np.full(3, 0.5), Delta_t=3.0, Vin=1.0
    )

    assert arrays.Volume_out() == pytest.approx(lists.Volume_out())
