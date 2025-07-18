"""
Tests for the SIHQUAL Excel file loading functionality.
"""
import os
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from mogestpy.quantity.Hydrodynamic.sihqual import SIHQUAL


class TestSIHQUALExcel:
    """Test suite for SIHQUAL Excel file loading."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary Excel file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.excel_path = os.path.join(self.temp_dir.name, 'test_sihqual.xlsx')
        self.inputs_dir = os.path.join(self.temp_dir.name, 'SIHQUALInputs')
        
        # Create inputs directory
        os.makedirs(self.inputs_dir, exist_ok=True)
        
        # Create test data
        self._create_test_excel_file()
        self._create_test_boundary_files()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def _create_test_excel_file(self):
        """Create a test Excel file with required sheets."""
        # Parameters sheet
        param_data = {
            'dx': [100.0],
            'dt': [10.0],
            'xf': [1000.0],
            'tf': [3600.0],
            'alpha': [0.6],
            'D': [5.0],
            'output_sections': [[0, 500, 1000]]
        }
        param_df = pd.DataFrame(param_data)
        
        # Data sheet
        x = np.linspace(0, 1000, 11)
        data = {
            'x': x,
            'b1': np.ones_like(x) * 10.0,
            'y1': np.ones_like(x) * 2.0,
            'm': np.zeros_like(x),
            'n': np.ones_like(x) * 0.03,
            'So': np.ones_like(x) * 0.001,
            'v1': np.ones_like(x) * 0.5,
            'kd': np.ones_like(x) * 0.0001,
            'ks': np.zeros_like(x),
            'c1': np.zeros_like(x),
            'cq': np.zeros_like(x)
        }
        data_df = pd.DataFrame(data)
        
        # Write to Excel file
        with pd.ExcelWriter(self.excel_path) as writer:
            param_df.to_excel(writer, sheet_name='Parameters', index=False)
            data_df.to_excel(writer, sheet_name='Data', index=False)
    
    def _create_test_boundary_files(self):
        """Create test boundary condition files."""
        # Time array
        time = np.array([0, 1800, 3600])
        
        # Flow boundary
        boundary_Q = pd.DataFrame({
            'time': time,
            '0': [10.0, 15.0, 10.0]
        })
        boundary_Q.to_csv(os.path.join(self.inputs_dir, 'boundary_Q.csv'), 
                          sep=';', index=False)
        
        # Water level boundary
        boundary_y = pd.DataFrame({
            'time': time,
            '1000': [2.0, 2.2, 2.0]
        })
        boundary_y.to_csv(os.path.join(self.inputs_dir, 'boundary_y.csv'), 
                          sep=';', index=False)
        
        # Concentration boundary
        boundary_c = pd.DataFrame({
            'time': time,
            '0': [0.0, 5.0, 0.0]
        })
        boundary_c.to_csv(os.path.join(self.inputs_dir, 'boundary_c.csv'), 
                          sep=';', index=False)
        
        # Lateral flow
        lateral_Q = pd.DataFrame({
            'time': time,
            ('500', '600'): [0.1, 0.2, 0.1]
        })
        lateral_Q.to_csv(os.path.join(self.inputs_dir, 'lateral_Q.csv'), 
                         sep=';', index=False)
        
        # Lateral concentration
        lateral_c = pd.DataFrame({
            'time': time,
            ('500', '600'): [1.0, 2.0, 1.0]
        })
        lateral_c.to_csv(os.path.join(self.inputs_dir, 'lateral_c.csv'), 
                         sep=';', index=False)
    
    def test_from_excel_loading(self):
        """Test loading a model from an Excel file."""
        # Load model from Excel
        model = SIHQUAL.from_excel(self.excel_path)
        
        # Check if basic parameters were loaded correctly
        assert model.dx == 100.0
        assert model.dt == 10.0
        assert model.xf == 1000.0
        assert model.tf == 3600.0
        assert model.alpha == 0.6
        assert model.dispersion_coef == 5.0
        
        # Check if geometry was loaded correctly
        assert np.all(model.b1 == 10.0)
        assert np.all(model.m == 0.0)
        assert np.all(model.n == 0.03)
        assert np.all(model.So == 0.001)
        
        # Check if initial conditions were loaded correctly
        assert np.all(model.y1 == 2.0)
        assert np.all(model.v1 == 0.5)
        assert np.all(model.c1 == 0.0)
        
        # Check if reaction parameters were loaded correctly
        assert np.all(model.Kd == 0.0001)
        assert np.all(model.Ks == 0.0)
        
        # Check if output sections were loaded correctly
        assert model.output_sections == [0, 5, 10]  # Indices for 0m, 500m, 1000m
        
        # Check if boundary conditions were loaded correctly
        assert model.index_Q == [0]
        assert model.index_y == [10]
        assert model.index_c == [0]
        
        # Check if lateral inflows were loaded correctly
        assert len(model.lQ_slices) == 1
        assert model.lQ_slices[0][0] == 5  # Index for 500m
        assert model.lQ_slices[0][1] == 6  # Index for 600m
    
    def test_excel_model_simulation(self):
        """Test running a simulation with a model loaded from Excel."""
        # Load model from Excel
        model = SIHQUAL.from_excel(self.excel_path)
        
        # Run a short simulation
        results = model.run(show_progress=False)
        
        # Check if results are returned
        assert isinstance(results, pd.DataFrame)
        assert not results.empty
        
        # Check if results have the expected structure
        expected_columns = pd.MultiIndex.from_product(
            [[0, 5, 10], ['Q', 'y', 'c']], names=['section', 'variables'])
        
        assert list(results.columns) == list(expected_columns)
        
        # Check if results have reasonable values
        assert np.all(results.loc[0, (0, 'Q')] > 0)  # Flow should be positive
        assert np.all(results.loc[0, (0, 'y')] > 0)  # Depth should be positive
    
    def test_excel_missing_files_handling(self):
        """Test handling of missing input files."""
        # Create a minimal Excel file without boundary files
        minimal_excel_path = os.path.join(self.temp_dir.name, 'minimal_test.xlsx')
        
        # Parameters sheet
        param_data = {
            'dx': [100.0],
            'dt': [10.0],
            'xf': [1000.0],
            'tf': [3600.0],
            'alpha': [0.6],
            'D': [5.0]
        }
        param_df = pd.DataFrame(param_data)
        
        # Data sheet
        x = np.linspace(0, 1000, 11)
        data = {
            'x': x,
            'b1': np.ones_like(x) * 10.0,
            'y1': np.ones_like(x) * 2.0,
            'm': np.zeros_like(x),
            'n': np.ones_like(x) * 0.03,
            'So': np.ones_like(x) * 0.001,
            'v1': np.ones_like(x) * 0.5,
            'kd': np.ones_like(x) * 0.0001,
            'ks': np.zeros_like(x),
            'c1': np.zeros_like(x),
            'cq': np.zeros_like(x)
        }
        data_df = pd.DataFrame(data)
        
        # Write to Excel file
        with pd.ExcelWriter(minimal_excel_path) as writer:
            param_df.to_excel(writer, sheet_name='Parameters', index=False)
            data_df.to_excel(writer, sheet_name='Data', index=False)
        
        # Load model from Excel - should not raise exceptions even with missing files
        model = SIHQUAL.from_excel(minimal_excel_path)
        
        # Check if basic parameters were loaded correctly
        assert model.dx == 100.0
        assert model.dt == 10.0
        assert model.xf == 1000.0
        assert model.tf == 3600.0
        
        # Check if geometry was loaded correctly
        assert np.all(model.b1 == 10.0)
        assert np.all(model.m == 0.0)
        assert np.all(model.n == 0.03)
        assert np.all(model.So == 0.001)
        
        # Boundary conditions should be empty
        assert model.index_Q == []
        assert model.index_y == []
        assert model.index_c == []
        
        # Should still be able to run a simulation
        results = model.run(show_progress=False)
        assert isinstance(results, pd.DataFrame)
    
    def test_excel_missing_sheet_handling(self):
        """Test handling of missing sheets in Excel file."""
        # Create an Excel file with only the Parameters sheet
        partial_excel_path = os.path.join(self.temp_dir.name, 'partial_test.xlsx')
        
        # Parameters sheet only
        param_data = {
            'dx': [100.0],
            'dt': [10.0],
            'xf': [1000.0],
            'tf': [3600.0],
            'alpha': [0.6],
            'D': [5.0]
        }
        param_df = pd.DataFrame(param_data)
        
        # Write to Excel file
        with pd.ExcelWriter(partial_excel_path) as writer:
            param_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        # Loading should raise an exception due to missing Data sheet
        with pytest.raises(Exception):
            SIHQUAL.from_excel(partial_excel_path)
    
    def test_excel_invalid_format_handling(self):
        """Test handling of invalid Excel file format."""
        # Create an Excel file with invalid format
        invalid_excel_path = os.path.join(self.temp_dir.name, 'invalid_test.xlsx')
        
        # Invalid Parameters sheet
        param_data = {
            'invalid_param': [100.0],
            'another_invalid': [10.0]
        }
        param_df = pd.DataFrame(param_data)
        
        # Write to Excel file
        with pd.ExcelWriter(invalid_excel_path) as writer:
            param_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        # Loading should raise an exception due to invalid format
        with pytest.raises(Exception):
            SIHQUAL.from_excel(invalid_excel_path)


if __name__ == "__main__":
    pytest.main(["-v", "test_sihqual_excel.py"])