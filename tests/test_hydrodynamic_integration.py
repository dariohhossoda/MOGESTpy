"""
Integration tests for hydrodynamic and hydrological modules using pytest.
"""
import pytest
from mogestpy.quantity.Hydrodynamic.saint_venant import TrapezoidalCrossSection, SaintVenant
from mogestpy.quantity.Hydrological.muskingum import Muskingum


class TestHydrodynamicIntegration:
    """Integration tests for hydrodynamic and hydrological modules."""

    @pytest.fixture
    def setup_data(self):
        """Set up test data."""
        # Create a sample hydrograph
        hydrograph = [10, 15, 25, 40, 60, 80, 70, 50, 35, 25, 20, 15, 10]

        # Create a trapezoidal cross-section
        cross_section = TrapezoidalCrossSection(10.0, 2.0, 1.0)

        # Define common parameters
        manning_n = 0.035
        slope = 0.001
        dt = 3600.0  # 1 hour in seconds
        dx = 1000.0  # 1 km in meters

        return {
            'hydrograph': hydrograph,
            'cross_section': cross_section,
            'manning_n': manning_n,
            'slope': slope,
            'dt': dt,
            'dx': dx
        }

    def test_normal_depth_consistency(self, setup_data):
        """Test consistency of normal depth calculations."""
        # Get test data
        hydrograph = setup_data['hydrograph']
        cross_section = setup_data['cross_section']
        manning_n = setup_data['manning_n']
        slope = setup_data['slope']

        # For each discharge in the hydrograph, calculate normal depth
        depths = []
        for discharge in hydrograph:
            depth = cross_section.normal_depth(discharge, manning_n, slope)
            depths.append(depth)

        # Verify depths increase with discharge
        for i in range(1, len(hydrograph)):
            if hydrograph[i] > hydrograph[i-1]:
                assert depths[i] > depths[i-1]
            elif hydrograph[i] < hydrograph[i-1]:
                assert depths[i] < depths[i-1]

    def test_muskingum_saint_venant_comparison(self, setup_data):
        """
        Compare Muskingum routing with Saint-Venant for simple cases.

        For mild slopes and subcritical flow, Muskingum should approximate
        the Saint-Venant equations reasonably well.
        """
        # Get test data
        hydrograph = setup_data['hydrograph']
        dt = setup_data['dt']

        # Route the hydrograph using Muskingum
        k = 3600.0  # 1 hour in seconds
        x = 0.2
        muskingum_outflow = Muskingum.downstream_routing(hydrograph, k, x, dt)

        # For Saint-Venant, we would need to implement the full numerical solution
        # Here we just check that the Muskingum result is physically reasonable

        # Verify peak attenuation
        inflow_peak = max(hydrograph)
        outflow_peak = max(muskingum_outflow)
        assert outflow_peak < inflow_peak

        # Verify peak delay
        inflow_peak_index = hydrograph.index(inflow_peak)
        outflow_peak_index = muskingum_outflow.index(outflow_peak)
        assert outflow_peak_index >= inflow_peak_index

        # Verify conservation of volume (within tolerance)
        inflow_volume = sum(hydrograph)
        outflow_volume = sum(muskingum_outflow)
        assert outflow_volume == pytest.approx(inflow_volume, rel=0.05)

    @pytest.mark.parametrize("dt,dx,expected_stable", [
        (3600, 5000, True),   # stable case
        (3600, 1000, None),   # potentially stable, depends on actual calculation
        (3600, 100, False)    # likely unstable
    ])
    def test_courant_stability(self, setup_data, dt, dx, expected_stable):
        """Test Courant stability condition for different parameters."""
        # Get test data
        cross_section = setup_data['cross_section']
        manning_n = setup_data['manning_n']
        slope = setup_data['slope']
        hydrograph = setup_data['hydrograph']

        # Use the peak discharge for worst-case scenario
        peak_discharge = max(hydrograph)

        # Create Saint-Venant model
        sv = SaintVenant(
            cross_section,
            peak_discharge,
            manning_n,
            slope,
            dt,
            dx
        )

        # Calculate Courant number
        courant = sv.courant()

        # Check stability condition
        is_stable = sv.courant_check()

        # Verify Courant number is positive
        assert courant > 0

        # Verify stability matches expectation based on Courant number
        if expected_stable is not None:
            assert is_stable == expected_stable

        # Always verify that stability check matches Courant condition
        assert is_stable == (courant <= 1)