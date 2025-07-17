"""
Unit tests for the Saint-Venant module using pytest.
"""
import math
import pytest
from mogestpy.quantity.Hydrodynamic.saint_venant import TrapezoidalCrossSection, SaintVenant, average


class TestTrapezoidalCrossSection:
    """Test cases for the TrapezoidalCrossSection class."""

    @pytest.fixture
    def cross_section(self):
        """Create a trapezoidal cross-section fixture."""
        # Create a trapezoidal cross-section with:
        # b = 5 (bottom width)
        # y = 2 (water depth)
        # m = 1.5 (side slope)
        return TrapezoidalCrossSection(5.0, 2.0, 1.5)

    def test_calculate_top_width(self, cross_section):
        """Test calculation of top width."""
        # Top width = b + 2*m*y = 5 + 2*1.5*2 = 11
        expected = 11.0
        assert cross_section.calculate_top_width() == pytest.approx(expected)
        assert cross_section.top_width == pytest.approx(expected)

    def test_wet_area(self, cross_section):
        """Test calculation of wet area."""
        # Wet area = b*y + m*y^2 = 5*2 + 1.5*2^2 = 10 + 6 = 16
        expected = 16.0
        assert cross_section.wet_area() == pytest.approx(expected)

    def test_wet_perimeter(self, cross_section):
        """Test calculation of wet perimeter."""
        # Wet perimeter = b + 2*y*sqrt(1+m^2) = 5 + 2*2*sqrt(1+1.5^2)
        expected = 5.0 + 4.0 * math.sqrt(1 + 2.25)
        assert cross_section.wet_perimeter() == pytest.approx(expected)

    def test_hydraulic_radius(self, cross_section):
        """Test calculation of hydraulic radius."""
        # Hydraulic radius = wet_area / wet_perimeter
        wet_area = 16.0
        wet_perimeter = 5.0 + 4.0 * math.sqrt(1 + 2.25)
        expected = wet_area / wet_perimeter
        assert cross_section.hydraulic_radius == pytest.approx(expected)

    def test_normal_depth(self, cross_section):
        """Test calculation of normal depth."""
        # Test with sample values
        discharge = 10.0  # m³/s
        manning_n = 0.03  # Manning's roughness coefficient
        slope = 0.001  # Channel slope

        # Calculate normal depth
        depth = cross_section.normal_depth(discharge, manning_n, slope)

        # Verify depth is positive and reasonable
        assert depth > 0
        assert depth < 5  # Should be less than 5m for these parameters

    def test_area_depth(self, cross_section):
        """Test calculation of depth from area."""
        # Test with a known area
        area = 16.0  # m²

        # Calculate depth
        depth = cross_section.area_depth(area)

        # Verify depth matches our setup (should be close to 2.0)
        assert depth == pytest.approx(2.0, abs=1e-5)


class TestSaintVenant:
    """Test cases for the SaintVenant class."""

    @pytest.fixture
    def saint_venant_model(self):
        """Create a Saint-Venant model fixture."""
        # Create a trapezoidal cross-section
        cross_section = TrapezoidalCrossSection(5.0, 2.0, 1.5)

        # Create a Saint-Venant model
        discharge = 10.0  # m³/s
        manning_n = 0.03
        slope = 0.001
        dt = 60.0  # seconds
        dx = 100.0  # meters
        return SaintVenant(
            cross_section,
            discharge,
            manning_n,
            slope,
            dt,
            dx
        )

    def test_courant(self, saint_venant_model):
        """Test calculation of Courant number."""
        # Calculate Courant number
        courant = saint_venant_model.courant()

        # Verify Courant number is positive
        assert courant > 0

    def test_courant_check(self):
        """Test Courant stability check."""
        # Create a trapezoidal cross-section
        cross_section = TrapezoidalCrossSection(5.0, 2.0, 1.5)
        discharge = 10.0
        manning_n = 0.03
        slope = 0.001

        # Test with a stable configuration
        stable_sv = SaintVenant(
            cross_section,
            discharge,
            manning_n,
            slope,
            1.0,  # small dt
            100.0  # large dx
        )
        assert stable_sv.courant_check() is True

        # Test with an unstable configuration
        unstable_sv = SaintVenant(
            cross_section,
            discharge,
            manning_n,
            slope,
            100.0,  # large dt
            1.0  # small dx
        )
        assert unstable_sv.courant_check() is False

    def test_quadratic_froude(self, saint_venant_model):
        """Test calculation of quadratic Froude number."""
        # Calculate quadratic Froude number
        froude_squared = saint_venant_model.quadratic_froude()

        # Verify Froude number is positive
        assert froude_squared > 0

    def test_friction_slope(self, saint_venant_model):
        """Test calculation of friction slope."""
        # Calculate friction slope
        friction_slope = saint_venant_model.friction_slope()

        # Verify friction slope is positive
        assert friction_slope > 0


def test_average():
    """Test the average function."""
    # Test with a valid list and index
    values = [1, 2, 3, 4, 5]
    assert average(values, 2) == 3.0  # Average of 1 and 5 is 3

    # Test with index at the beginning (should return None)
    assert average(values, 0) is None

    # Test with index at the end (should return None)
    assert average(values, 4) is None
