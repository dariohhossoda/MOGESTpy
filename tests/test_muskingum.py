"""
Unit tests for the Muskingum routing module using pytest.
"""

import pytest
import numpy as np
import pandas as pd
from mogestpy.quantity.hydrological.muskingum import Muskingum, MuskingumCunge


class TestMuskingum:
    """Test cases for the Muskingum class."""

    @pytest.fixture
    def setup_data(self):
        """Set up test data."""
        # Create sample inflow hydrograph
        inflow = [10, 20, 40, 60, 80, 70, 50, 40, 30, 20, 15, 10]

        # Define Muskingum parameters
        k = 24.0  # storage coefficient (hours)
        x = 0.3  # weighting factor
        m = 1.0  # exponent (linear case)
        dt = 24.0  # time step (hours)

        return {"inflow": inflow, "k": k, "x": x, "m": m, "dt": dt}

    def test_downstream_routing_linear(self, setup_data):
        """Test linear downstream routing."""
        # Get test data
        inflow = setup_data["inflow"]
        k = setup_data["k"]
        x = setup_data["x"]
        dt = setup_data["dt"]

        # Calculate outflow using Muskingum method
        outflow = Muskingum.downstream_routing(inflow, k, x, dt)

        # Verify outflow has same length as inflow
        assert len(outflow) == len(inflow)

        # Verify first value of outflow equals first value of inflow
        assert outflow[0] == inflow[0]

        # Verify outflow peak is attenuated and delayed compared to inflow
        inflow_peak_index = np.argmax(inflow)
        outflow_peak_index = np.argmax(outflow)

        # Peak should be delayed
        assert outflow_peak_index >= inflow_peak_index

        # Peak should be attenuated
        assert outflow[outflow_peak_index] < inflow[inflow_peak_index]

        # Verify mass conservation (sum of outflow should be close to sum of inflow)
        assert sum(outflow) == pytest.approx(sum(inflow), rel=0.05)

    def test_downstream_fork_nonlinear(self, setup_data):
        """Test non-linear downstream routing."""
        # Get test data
        inflow = setup_data["inflow"]
        k = setup_data["k"]
        x = setup_data["x"]
        dt = setup_data["dt"]

        # Use non-linear exponent
        m = 1.2

        # Calculate outflow using non-linear Muskingum method
        outflow = Muskingum.downstream_fork(k, x, m, dt, inflow)

        # Verify outflow has same length as inflow
        assert len(outflow) == len(inflow)

        # Verify first value of outflow equals first value of inflow
        assert outflow[0] == inflow[0]

        # Verify all outflow values are non-negative
        for value in outflow:
            assert value >= 0

    def test_parameter_bounds(self, setup_data):
        """Test behavior with extreme parameter values."""
        # Get test data
        inflow = setup_data["inflow"]
        k = setup_data["k"]
        dt = setup_data["dt"]

        # Test with x close to 0 (reservoir-type behavior)
        x_small = 0.01
        outflow_x_small = Muskingum.downstream_routing(inflow, k, x_small, dt)

        # Test with x close to 0.5 (pure translation)
        x_large = 0.49
        outflow_x_large = Muskingum.downstream_routing(inflow, k, x_large, dt)

        # Verify both produce valid results
        assert len(outflow_x_small) == len(inflow)
        assert len(outflow_x_large) == len(inflow)

        # With small x, attenuation should be greater
        peak_small = max(outflow_x_small)
        peak_large = max(outflow_x_large)
        assert peak_small < peak_large

    def test_conservation_of_mass(self, setup_data):
        """Test conservation of mass in routing."""
        # Get test data
        inflow = setup_data["inflow"]
        k = setup_data["k"]
        x = setup_data["x"]
        dt = setup_data["dt"]

        # For both linear and non-linear methods
        methods = [
            lambda: Muskingum.downstream_routing(inflow, k, x, dt),
            lambda: Muskingum.downstream_fork(k, x, 1.2, dt, inflow),
        ]

        for method in methods:
            outflow = method()
            # Sum of outflow should be close to sum of inflow (within 5%)
            assert sum(outflow) == pytest.approx(sum(inflow), rel=0.05)


class TestMuskingumCunge:
    """Test cases for the MuskingumCunge class."""

    @pytest.fixture
    def reach(self):
        """Reach of 60 km draining 2500 km² (MGB regionalization)."""
        area = 2500.0
        return {
            "length": 60_000.0,
            "q_ref": area**0.7 / 3,
            "manning": 0.035,
            "width": 1.5 * area**0.5,
            "slope": 2e-4,
        }

    @pytest.fixture
    def flood(self):
        """Daily flood hydrograph (m³/s)."""
        return [50, 50, 120, 400, 650, 500, 320, 210, 140, 100, 80, 65, 55, 50]

    def test_parameters_keep_coefficients_non_negative(self, reach):
        """Coefficients are non-negative, sum to one and X stays in range."""
        params = MuskingumCunge.parameters(dt=3600.0, **reach)

        coefficients = [params["c0"], params["c1"], params["c2"]]
        assert min(coefficients) >= 0
        assert sum(coefficients) == pytest.approx(1.0)
        assert 0.0 <= params["x"] <= 0.5
        assert params["dt"] <= 3600.0
        assert params["n_subreaches"] * params["dx"] == pytest.approx(reach["length"])

    def test_steady_inflow_is_unchanged(self, reach):
        """A constant inflow leaves the reach unchanged."""
        outflow = MuskingumCunge.downstream_routing([80.0] * 10, **reach)

        assert outflow == pytest.approx([80.0] * 10)

    def test_flood_is_attenuated_and_conserved(self, reach, flood):
        """The flood peak is attenuated while the volume is conserved."""
        outflow = MuskingumCunge.downstream_routing(flood, **reach)

        assert len(outflow) == len(flood)
        assert outflow.max() < max(flood)
        assert outflow.sum() == pytest.approx(sum(flood), rel=0.01)

    def test_short_reach_is_routed_on_a_daily_step(self, flood):
        """A reach shorter than a day of travel is still routed."""
        outflow = MuskingumCunge.downstream_routing(
            flood, length=3000.0, q_ref=80.0, manning=0.035, width=40.0, slope=5e-4
        )

        assert not np.allclose(outflow, flood)
        assert outflow == pytest.approx(flood, rel=0.05)

    def test_single_step_matches_linear_muskingum(self, flood):
        """With one sub-reach and no sub-steps it equals linear Muskingum."""
        area = 80.0
        reach = {
            "length": 3000.0,
            "q_ref": area**0.7 / 3,
            "manning": 0.035,
            "width": 1.5 * area**0.5,
            "slope": 5e-4,
        }
        params = MuskingumCunge.parameters(dt=3600.0, **reach)
        assert params["n_subreaches"] == 1
        assert params["dt"] == 3600.0

        outflow = MuskingumCunge.downstream_routing(
            flood, dt=3600.0, sub_dt=3600.0, **reach
        )
        expected = Muskingum.downstream_routing(
            flood, params["k"], params["x"], params["dt"]
        )

        assert outflow == pytest.approx(expected)

    def test_rejects_non_positive_reach_characteristics(self, reach):
        """Reach characteristics must be positive."""
        reach["slope"] = 0.0

        with pytest.raises(ValueError, match="slope"):
            MuskingumCunge.parameters(dt=3600.0, **reach)

    def test_router_reads_reach_from_row(self, reach, flood):
        """The router reads the reach from a parameter row."""
        route = MuskingumCunge.router()
        row = pd.Series(reach)

        assert route(np.asarray(flood, dtype=float), row) == pytest.approx(
            MuskingumCunge.downstream_routing(flood, **reach)
        )

    def test_router_passes_flow_through_zero_length_reach(self, reach, flood):
        """A reach with zero length passes the flow through."""
        route = MuskingumCunge.router()
        row = pd.Series({**reach, "length": 0.0})

        assert route(np.asarray(flood, dtype=float), row) == pytest.approx(flood)
