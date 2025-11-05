"""
Unit tests for the Muskingum routing module using pytest.
"""
import pytest
import numpy as np
from mogestpy.quantity.Hydrological.muskingum import Muskingum


class TestMuskingum:
    """Test cases for the Muskingum class."""

    @pytest.fixture
    def setup_data(self):
        """Set up test data."""
        # Create sample inflow hydrograph
        inflow = [10, 20, 40, 60, 80, 70, 50, 40, 30, 20, 15, 10]
        
        # Define Muskingum parameters
        k = 24.0  # storage coefficient (hours)
        x = 0.3   # weighting factor
        m = 1.0   # exponent (linear case)
        dt = 24.0  # time step (hours)
        
        return {
            'inflow': inflow,
            'k': k,
            'x': x,
            'm': m,
            'dt': dt
        }

    def test_downstream_routing_linear(self, setup_data):
        """Test linear downstream routing."""
        # Get test data
        inflow = setup_data['inflow']
        k = setup_data['k']
        x = setup_data['x']
        dt = setup_data['dt']
        
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
        inflow = setup_data['inflow']
        k = setup_data['k']
        x = setup_data['x']
        dt = setup_data['dt']
        
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

    def test_upstream_fork_nonlinear(self, setup_data):
        """Test non-linear upstream routing."""
        # Get test data
        inflow = setup_data['inflow']
        k = setup_data['k']
        x = setup_data['x']
        m = setup_data['m']
        dt = setup_data['dt']
        
        # First generate downstream values
        downstream = Muskingum.downstream_fork(k, x, m, dt, inflow)
        
        # Then route upstream
        upstream = Muskingum.upstream_fork(k, x, m, dt, downstream)
        
        # Verify upstream has same length as downstream
        assert len(upstream) == len(downstream)
        
        # Verify last value of upstream equals last value of downstream
        assert upstream[-1] == downstream[-1]
        
        # Verify all upstream values are non-negative
        for value in upstream:
            assert value >= 0
        
        # For linear case (m=1), upstream should be close to original inflow
        if m == 1.0:
            # Allow some numerical error
            for i in range(len(inflow)):
                assert upstream[i] == pytest.approx(inflow[i], rel=0.1)

    def test_parameter_bounds(self, setup_data):
        """Test behavior with extreme parameter values."""
        # Get test data
        inflow = setup_data['inflow']
        k = setup_data['k']
        dt = setup_data['dt']
        
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
        inflow = setup_data['inflow']
        k = setup_data['k']
        x = setup_data['x']
        dt = setup_data['dt']
        
        # For both linear and non-linear methods
        methods = [
            lambda: Muskingum.downstream_routing(inflow, k, x, dt),
            lambda: Muskingum.downstream_fork(k, x, 1.2, dt, inflow)
        ]
        
        for method in methods:
            outflow = method()
            # Sum of outflow should be close to sum of inflow (within 5%)
            assert sum(outflow) == pytest.approx(sum(inflow), rel=0.05)