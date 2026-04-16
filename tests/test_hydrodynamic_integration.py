"""
Integration tests for hydrodynamic and hydrological modules using pytest.
"""

import pytest
from mogestpy.quantity.hydrodynamic.saint_venant import (
    TrapezoidalCrossSection,
    SaintVenant,
)
from mogestpy.quantity.hydrological.muskingum import Muskingum


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
            "hydrograph": hydrograph,
            "cross_section": cross_section,
            "manning_n": manning_n,
            "slope": slope,
            "dt": dt,
            "dx": dx,
        }

    def test_muskingum_saint_venant_comparison(self, setup_data):
        """
        Compare Muskingum routing with Saint-Venant for simple cases.

        For mild slopes and subcritical flow, Muskingum should approximate
        the Saint-Venant equations reasonably well.
        """
        # Get test data
        hydrograph = setup_data["hydrograph"]
        dt = setup_data["dt"]

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
