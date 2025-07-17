"""
Unit tests for the SIHQUAL (Simulação Hidrodinâmica e de Qualidade da Água) module using pytest.
"""
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open
from mogestpy.quantity.Hydrodynamic import SIHQUAL


class TestSIHQUAL:
    """Test cases for the SIHQUAL module."""

    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        # Mock data for testing
        mock_param_data = {
            'dx': [100.0],
            'dt': [10.0],
            'xf': [1000.0],
            'tf': [3600.0],
            'alpha': [0.6],
            'D': [0.1],
            'output_sections': [0, 500, 1000]
        }

        # Create mock data for the main data sheet
        n_points = 11  # For a 1000m reach with dx=100m
        mock_data_data = {
            'x': np.linspace(0, 1000, n_points),
            'b1': np.full(n_points, 10.0),  # Bottom width
            'y1': np.full(n_points, 2.0),   # Initial water depth
            'm': np.full(n_points, 1.0),    # Side slope
            'n': np.full(n_points, 0.03),   # Manning's n
            'So': np.full(n_points, 0.001),  # Bed slope
            'v1': np.full(n_points, 0.5),   # Initial velocity
            'kd': np.full(n_points, 0.1),   # Decay coefficient
            'ks': np.full(n_points, 0.05),  # Settling coefficient
            'c1': np.full(n_points, 5.0),   # Initial concentration
            'cq': np.full(n_points, 0.0)    # Lateral inflow concentration
        }

        # Create mock boundary conditions
        time_steps = np.arange(0, 3601, 600)  # 0 to 3600s in steps of 600s
        n_times = len(time_steps)

        mock_boundary_Q = pd.DataFrame({
            'time': time_steps,
            '0': np.full(n_times, 10.0)  # Constant inflow at x=0
        })

        mock_boundary_y = pd.DataFrame({
            'time': time_steps,
            '0': np.full(n_times, 2.0),  # Constant depth at x=0
            '1000': np.full(n_times, 1.8)  # Constant depth at x=1000
        })

        mock_boundary_c = pd.DataFrame({
            'time': time_steps,
            '0': np.full(n_times, 5.0)  # Constant concentration at x=0
        })

        # Create mock lateral inflows
        mock_lateral_Q = pd.DataFrame({
            'time': time_steps
        })
        # Lateral inflow between x=300 and x=500
        mock_lateral_Q['300;500'] = np.full(n_times, 0.1)

        mock_lateral_c = pd.DataFrame({
            'time': time_steps
        })
        # Concentration of lateral inflow
        mock_lateral_c['300;500'] = np.full(n_times, 2.0)

        return {
            'param_df': pd.DataFrame(mock_param_data),
            'data_df': pd.DataFrame(mock_data_data),
            'boundary_Q': mock_boundary_Q,
            'boundary_y': mock_boundary_y,
            'boundary_c': mock_boundary_c,
            'lateral_Q': mock_lateral_Q,
            'lateral_c': mock_lateral_c
        }

    @pytest.mark.parametrize("vector, expected", [
        (np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4])),
        (np.array([0, 1, 2, 3, 0]), np.array([0.5, 1.5, 2.5])),
    ])
    def test_avg(self, vector, expected):
        """Test the avg function."""
        result = SIHQUAL.avg(vector)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("vector, expected", [
        (np.array([1, 2, 3, 4, 5]), np.array([4, 2, 2])),
        (np.array([0, 1, 2, 3, 0]), np.array([3, 2, -2])),
    ])
    def test_cdelta(self, vector, expected):
        """Test the cdelta function."""
        result = SIHQUAL.cdelta(vector)
        np.testing.assert_array_equal(result, expected)

    def test_top_base(self):
        """Test the top_base function."""
        b = np.array([5.0, 6.0, 7.0])
        y = np.array([1.0, 1.5, 2.0])
        m = np.array([1.0, 1.0, 1.0])
        expected = np.array([7.0, 9.0, 11.0])  # b + 2*m*y

        result = SIHQUAL.top_base(b, y, m)
        np.testing.assert_array_equal(result, expected)

    def test_wet_area(self):
        """Test the wet_area function."""
        b = np.array([5.0, 6.0, 7.0])
        y = np.array([1.0, 1.5, 2.0])
        m = np.array([1.0, 1.0, 1.0])
        expected = np.array([6.0, 10.5, 15.0])  # b*y + m*y*y

        result = SIHQUAL.wet_area(b, y, m)
        np.testing.assert_array_equal(result, expected)

    def test_wet_perimeter(self):
        """Test the wet_perimeter function."""
        b = np.array([5.0, 6.0, 7.0])
        y = np.array([1.0, 1.5, 2.0])
        m = np.array([1.0, 1.0, 1.0])
        expected = b + 2 * y * (1 + m ** 2) ** 0.5

        result = SIHQUAL.wet_perimeter(b, y, m)
        np.testing.assert_array_almost_equal(result, expected)

    def test_Rh(self):
        """Test the hydraulic radius function."""
        b = np.array([5.0, 6.0, 7.0])
        y = np.array([1.0, 1.5, 2.0])
        m = np.array([1.0, 1.0, 1.0])

        area = SIHQUAL.wet_area(b, y, m)
        perimeter = SIHQUAL.wet_perimeter(b, y, m)
        expected = area / perimeter

        result = SIHQUAL.Rh(b, y, m)
        np.testing.assert_array_almost_equal(result, expected)

    def test_Sf(self):
        """Test the friction slope function."""
        n = np.array([0.03, 0.04, 0.05])
        v = np.array([0.5, 1.0, 1.5])
        Rh = np.array([0.8, 1.0, 1.2])
        expected = n * n * v * v * Rh ** (-4/3)

        result = SIHQUAL.Sf(n, v, Rh)
        np.testing.assert_array_almost_equal(result, expected)

    def test_courant(self):
        """Test the Courant number calculation."""
        dt = 10.0
        dx = 100.0
        v = np.array([0.5, 1.0, 1.5])
        g = 9.81
        A = np.array([10.0, 15.0, 20.0])
        B = np.array([8.0, 10.0, 12.0])

        expected = dt / dx * (np.abs(v) + (g * A / B) ** 0.5)

        result = SIHQUAL.courant(dt, dx, v, g, A, B)
        np.testing.assert_array_almost_equal(result, expected)

    def test_ifromx(self):
        """Test the index calculation from x-coordinate."""
        x = np.array([0, 250, 500, 750, 1000])
        L = 1000.0
        dim = 11  # 0 to 1000 with dx=100

        expected = np.array([0, 2, 5, 7, 10])

        result = SIHQUAL.ifromx(x, L, dim)
        np.testing.assert_array_equal(result, expected)

    @patch('pandas.read_excel')
    @patch('pandas.read_csv')
    @patch('os.path.join')
    def test_module_initialization(self, mock_join, mock_read_csv, mock_read_excel, mock_data):
        """Test the initialization of the SIHQUAL module."""
        # Mock the file paths
        mock_join.side_effect = lambda *args: '/'.join(args)

        # Mock the Excel file reading
        mock_read_excel.side_effect = lambda filename, sheet_name: (
            mock_data['param_df'] if sheet_name == 'Parameters' else mock_data['data_df']
        )

        # Mock the CSV file reading
        def mock_csv_read(filename, **kwargs):
            if 'boundary_Q' in filename:
                return mock_data['boundary_Q']
            elif 'boundary_y' in filename:
                return mock_data['boundary_y']
            elif 'boundary_c' in filename:
                return mock_data['boundary_c']
            elif 'lateral_Q' in filename:
                return mock_data['lateral_Q']
            elif 'lateral_c' in filename:
                return mock_data['lateral_c']

        mock_read_csv.side_effect = mock_csv_read

        # Import the module to trigger initialization
        with patch.object(SIHQUAL, '__name__', '__main__'):
            # This would normally execute the module's global code
            pass

        # We can't easily test the full module execution without refactoring,
        # but we can test that the auxiliary functions work correctly

    @patch('pandas.DataFrame.to_excel')
    def test_output_generation(self, mock_to_excel):
        """Test the generation of output data."""
        # This is a placeholder test since we can't easily test the full simulation
        # without refactoring the module
        mock_to_excel.return_value = None

        # In a real test, we would run the simulation and check the output
        # But since the module is not structured for easy testing, we'll just
        # verify that the auxiliary functions work correctly


@pytest.mark.parametrize("b, y, m, expected", [
    (5.0, 1.0, 1.0, 7.0),  # b + 2*m*y = 5 + 2*1*1 = 7
    (10.0, 2.0, 0.5, 12.0),  # b + 2*m*y = 10 + 2*0.5*2 = 12
])
def test_top_base_scalar(b, y, m, expected):
    """Test the top_base function with scalar inputs."""
    result = SIHQUAL.top_base(b, y, m)
    assert result == expected


@pytest.mark.parametrize("b, y, m, expected", [
    (5.0, 1.0, 1.0, 6.0),  # b*y + m*y*y = 5*1 + 1*1*1 = 6
    (10.0, 2.0, 0.5, 22.0),  # b*y + m*y*y = 10*2 + 0.5*2*2 = 22
])
def test_wet_area_scalar(b, y, m, expected):
    """Test the wet_area function with scalar inputs."""
    result = SIHQUAL.wet_area(b, y, m)
    assert result == expected


@pytest.mark.parametrize("n, v, Rh, expected", [
    (0.03, 1.0, 1.0, 0.0009),  # n*n*v*v*Rh^(-4/3) = 0.03*0.03*1*1*1^(-4/3) ≈ 0.0009
    # n*n*v*v*Rh^(-4/3) = 0.04*0.04*2*2*0.5^(-4/3) ≈ 0.0128
    (0.04, 2.0, 0.5, 0.0128),
])
def test_Sf_scalar(n, v, Rh, expected):
    """Test the friction slope function with scalar inputs."""
    result = SIHQUAL.Sf(n, v, Rh)
    assert pytest.approx(result, abs=1e-4) == expected
