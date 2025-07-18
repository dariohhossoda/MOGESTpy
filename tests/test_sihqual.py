"""
Tests for the SIHQUAL module.
"""
import os
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from mogestpy.quantity.Hydrodynamic.sihqual import SIHQUAL


class TestSIHQUAL:
    """Test suite for SIHQUAL class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Basic parameters for a simple test case
        self.dx = 100.0  # m
        self.dt = 10.0   # s
        self.xf = 1000.0  # m
        self.tf = 3600.0  # s (1 hour)
        
        # Create model instance
        self.model = SIHQUAL(
            dx=self.dx,
            dt=self.dt,
            xf=self.xf,
            tf=self.tf,
            alpha=0.6,
            dispersion_coef=5.0
        )
        
        # Number of spatial points
        self.dim = int(self.xf / self.dx) + 1
        
        # Set up geometry
        b1 = np.ones(self.dim) * 10.0  # 10m wide channel
        m = np.ones(self.dim) * 0.0    # Rectangular channel
        n = np.ones(self.dim) * 0.03   # Manning's n
        So = np.ones(self.dim) * 0.001  # Slope
        
        self.model.set_geometry(b1, m, n, So)
        
        # Set initial conditions
        y1 = np.ones(self.dim) * 2.0   # 2m depth
        v1 = np.ones(self.dim) * 0.5   # 0.5 m/s velocity
        c1 = np.zeros(self.dim)        # Zero initial concentration
        
        self.model.set_initial_conditions(y1, v1, c1)
        
        # Set reaction parameters
        Kd = np.ones(self.dim) * 0.0001  # Decay coefficient
        Ks = np.zeros(self.dim)          # No source
        
        self.model.set_reaction_parameters(Kd, Ks)
        
        # Set output sections
        self.model.set_output_sections([0.0, 500.0, 1000.0])
    
    def test_initialization(self):
        """Test model initialization."""
        assert self.model.dx == self.dx
        assert self.model.dt == self.dt
        assert self.model.xf == self.xf
        assert self.model.tf == self.tf
        assert self.model.dim == self.dim
        assert len(self.model.y1) == self.dim
        assert len(self.model.v1) == self.dim
        assert len(self.model.c1) == self.dim
    
    def test_geometry_setup(self):
        """Test geometry setup."""
        assert np.all(self.model.b1 == 10.0)
        assert np.all(self.model.m == 0.0)
        assert np.all(self.model.n == 0.03)
        assert np.all(self.model.So == 0.001)
    
    def test_initial_conditions(self):
        """Test initial conditions setup."""
        assert np.all(self.model.y1 == 2.0)
        assert np.all(self.model.v1 == 0.5)
        assert np.all(self.model.c1 == 0.0)
    
    def test_reaction_parameters(self):
        """Test reaction parameters setup."""
        assert np.all(self.model.Kd == 0.0001)
        assert np.all(self.model.Ks == 0.0)
    
    def test_output_sections(self):
        """Test output sections setup."""
        expected = [0, 5, 10]  # Indices for 0m, 500m, 1000m
        assert self.model.output_sections == expected
    
    def test_hydraulic_functions(self):
        """Test hydraulic calculation functions."""
        b = np.array([10.0])
        y = np.array([2.0])
        m = np.array([0.0])
        
        # Test wet area
        area = self.model._wet_area(b, y, m)
        assert area[0] == 20.0  # 10m * 2m
        
        # Test top width
        width = self.model._top_width(b, y, m)
        assert width[0] == 10.0  # 10m + 2*0*2m
        
        # Test wet perimeter
        perimeter = self.model._wet_perimeter(b, y, m)
        assert perimeter[0] == 14.0  # 10m + 2*2m
        
        # Test hydraulic radius
        radius = self.model._hydraulic_radius(b, y, m)
        assert radius[0] == pytest.approx(20.0/14.0)  # area/perimeter
    
    def test_boundary_conditions(self):
        """Test setting boundary conditions."""
        # Create simple boundary conditions
        boundary_positions = {
            'Q': [0.0],
            'y': [1000.0],
            'c': [0.0]
        }
        
        time_array = np.array([0, 1800, 3600])  # 0, 30min, 60min
        
        q_values = np.array([[10.0, 20.0, 10.0]])  # m³/s
        y_values = np.array([[2.0, 2.5, 2.0]])    # m
        c_values = np.array([[0.0, 5.0, 0.0]])    # mg/L
        
        boundary_data = {
            'Q': {'time': time_array, 'values': q_values},
            'y': {'time': time_array, 'values': y_values},
            'c': {'time': time_array, 'values': c_values}
        }
        
        self.model.set_boundary_conditions(boundary_positions, boundary_data)
        
        # Check if boundary conditions were set correctly
        assert self.model.index_Q == [0]
        assert self.model.index_y == [10]
        assert self.model.index_c == [0]
        
        assert np.array_equal(self.model.t_arr_Q, time_array)
        assert np.array_equal(self.model.t_arr_y, time_array)
        assert np.array_equal(self.model.t_arr_c, time_array)
        
        assert np.array_equal(self.model.bQ_data, q_values)
        assert np.array_equal(self.model.by_data, y_values)
        assert np.array_equal(self.model.bc_data, c_values)
    
    def test_lateral_inflows(self):
        """Test setting lateral inflows."""
        # Create simple lateral inflows
        lateral_Q_segments = [(200.0, 300.0), (600.0, 700.0)]
        
        time_array = np.array([0, 1800, 3600])  # 0, 30min, 60min
        
        q_values = np.array([
            [0.1, 0.2, 0.1],  # m³/s per meter for segment 1
            [0.2, 0.4, 0.2]   # m³/s per meter for segment 2
        ])
        
        lateral_Q_data = {
            'time': time_array,
            'values': q_values
        }
        
        self.model.set_lateral_inflows(lateral_Q_segments, lateral_Q_data)
        
        # Check if lateral inflows were set correctly
        expected_slices = [(2, 3), (6, 7)]  # Indices for the segments
        
        assert len(self.model.lQ_slices) == 2
        assert self.model.lQ_slices == expected_slices
        
        assert np.array_equal(self.model.t_arr_lQ, time_array)
        assert np.array_equal(self.model.lQ_data, q_values)
    
    def test_short_simulation_run(self):
        """Test a short simulation run."""
        # Set a shorter simulation time for testing
        self.model.tf = 60.0  # 1 minute
        
        # Set simple boundary conditions
        boundary_positions = {
            'Q': [0.0],
            'y': [1000.0]
        }
        
        time_array = np.array([0, 30, 60])  # 0, 30s, 60s
        
        q_values = np.array([[10.0, 10.0, 10.0]])  # m³/s
        y_values = np.array([[2.0, 2.0, 2.0]])    # m
        
        boundary_data = {
            'Q': {'time': time_array, 'values': q_values},
            'y': {'time': time_array, 'values': y_values}
        }
        
        self.model.set_boundary_conditions(boundary_positions, boundary_data)
        
        # Run the simulation
        results = self.model.run(show_progress=False)
        
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
    
    def test_courant_stability(self):
        """Test Courant stability check."""
        # Create a model with parameters that would violate stability
        unstable_model = SIHQUAL(
            dx=10.0,    # Small dx
            dt=10.0,    # Large dt
            xf=100.0,
            tf=100.0
        )
        
        # Set up geometry
        b1 = np.ones(11) * 10.0
        m = np.ones(11) * 0.0
        n = np.ones(11) * 0.03
        So = np.ones(11) * 0.001
        
        unstable_model.set_geometry(b1, m, n, So)
        
        # Set initial conditions with high velocity
        y1 = np.ones(11) * 2.0
        v1 = np.ones(11) * 5.0  # High velocity to trigger instability
        c1 = np.zeros(11)
        
        unstable_model.set_initial_conditions(y1, v1, c1)
        unstable_model.set_output_sections([0.0, 50.0, 100.0])
        
        # Check if stability error is raised
        with pytest.raises(ValueError, match="Stability error"):
            unstable_model.run(show_progress=False)
    
    def test_scalar_inputs(self):
        """Test model with scalar inputs instead of arrays."""
        # Create a new model
        model = SIHQUAL(
            dx=100.0,
            dt=10.0,
            xf=1000.0,
            tf=3600.0
        )
        
        # Set geometry with scalar values
        model.set_geometry(
            bottom_width=10.0,
            side_slope=0.0,
            manning_coef=0.03,
            bed_slope=0.001
        )
        
        # Set initial conditions with scalar values
        model.set_initial_conditions(
            initial_depth=2.0,
            initial_velocity=0.5,
            initial_concentration=0.0
        )
        
        # Set reaction parameters with scalar values
        model.set_reaction_parameters(
            decay_coef=0.0001,
            source_coef=0.0
        )
        
        # Check if arrays were properly initialized
        assert np.all(model.b1 == 10.0)
        assert np.all(model.m == 0.0)
        assert np.all(model.n == 0.03)
        assert np.all(model.So == 0.001)
        assert np.all(model.y1 == 2.0)
        assert np.all(model.v1 == 0.5)
        assert np.all(model.c1 == 0.0)
        assert np.all(model.Kd == 0.0001)
        assert np.all(model.Ks == 0.0)
    
    def test_partial_array_inputs(self):
        """Test model with partial array inputs that need interpolation."""
        # Create a new model
        model = SIHQUAL(
            dx=100.0,
            dt=10.0,
            xf=1000.0,
            tf=3600.0
        )
        
        # Set geometry with partial arrays (only 3 points)
        x_partial = np.array([0.0, 500.0, 1000.0])
        b_partial = np.array([10.0, 15.0, 20.0])  # Varying width
        m_partial = np.array([0.0, 0.5, 1.0])     # Varying side slope
        n_partial = np.array([0.03, 0.04, 0.05])  # Varying Manning's n
        So_partial = np.array([0.002, 0.001, 0.0005])  # Varying slope
        
        model.set_geometry(
            bottom_width=b_partial,
            side_slope=m_partial,
            manning_coef=n_partial,
            bed_slope=So_partial
        )
        
        # Check if arrays were properly interpolated
        # For x=0, should be b=10.0
        assert model.b1[0] == pytest.approx(10.0)
        # For x=500, should be b=15.0
        assert model.b1[5] == pytest.approx(15.0)
        # For x=1000, should be b=20.0
        assert model.b1[10] == pytest.approx(20.0)
        
        # Check intermediate values (should be linearly interpolated)
        assert model.b1[2] > 10.0 and model.b1[2] < 15.0
        assert model.m[2] > 0.0 and model.m[2] < 0.5
        assert model.n[2] > 0.03 and model.n[2] < 0.04
        assert model.So[2] < 0.002 and model.So[2] > 0.001
    
    def test_trapezoidal_channel(self):
        """Test model with a trapezoidal channel."""
        # Create a new model
        model = SIHQUAL(
            dx=100.0,
            dt=10.0,
            xf=1000.0,
            tf=3600.0
        )
        
        # Set geometry for a trapezoidal channel
        model.set_geometry(
            bottom_width=10.0,
            side_slope=2.0,  # 2:1 side slope (horizontal:vertical)
            manning_coef=0.03,
            bed_slope=0.001
        )
        
        # Set initial conditions
        model.set_initial_conditions(
            initial_depth=2.0,
            initial_velocity=0.5,
            initial_concentration=0.0
        )
        
        # Calculate expected values
        b = 10.0
        y = 2.0
        m = 2.0
        
        # Expected top width: b + 2*m*y = 10 + 2*2*2 = 18
        expected_top_width = b + 2 * m * y
        
        # Expected wet area: b*y + m*y^2 = 10*2 + 2*2^2 = 20 + 8 = 28
        expected_wet_area = b * y + m * y * y
        
        # Expected wet perimeter: b + 2*y*sqrt(1+m^2) = 10 + 2*2*sqrt(1+4) = 10 + 4*sqrt(5)
        expected_wet_perimeter = b + 2 * y * np.sqrt(1 + m * m)
        
        # Test hydraulic calculations
        top_width = model._top_width(np.array([b]), np.array([y]), np.array([m]))[0]
        wet_area = model._wet_area(np.array([b]), np.array([y]), np.array([m]))[0]
        wet_perimeter = model._wet_perimeter(np.array([b]), np.array([y]), np.array([m]))[0]
        
        assert top_width == pytest.approx(expected_top_width)
        assert wet_area == pytest.approx(expected_wet_area)
        assert wet_perimeter == pytest.approx(expected_wet_perimeter)
    
    def test_save_results(self):
        """Test saving simulation results."""
        # Create a simple DataFrame for testing
        sections = [0, 5, 10]
        variables = ['Q', 'y', 'c']
        cols = pd.MultiIndex.from_product([sections, variables], names=['section', 'variables'])
        t_index = pd.Index([0, 1], name='t')
        
        # Create sample data
        data = np.array([
            [10.0, 2.0, 0.0, 9.0, 1.9, 0.1, 8.0, 1.8, 0.2],  # Day 0
            [11.0, 2.1, 0.1, 10.0, 2.0, 0.2, 9.0, 1.9, 0.3]  # Day 1
        ])
        
        df = pd.DataFrame(data, columns=cols, index=t_index)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            temp_filename = tmp.name
        
        try:
            # Test saving
            result = self.model.save_results(df, temp_filename)
            assert result is True
            
            # Check if file exists and can be read
            assert os.path.exists(temp_filename)
            
            # Try to read it back
            df_read = pd.read_excel(temp_filename, index_col=0, header=[0, 1])
            
            # Check if data is preserved
            assert df_read.shape == df.shape
            
            # Check some values
            assert df_read.loc[0, (0, 'Q')] == pytest.approx(10.0)
            assert df_read.loc[1, (10, 'c')] == pytest.approx(0.3)
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_auto_step_adjustment(self):
        """Test automatic time step adjustment."""
        # Create a model with auto_step enabled
        model = SIHQUAL(
            dx=100.0,
            dt=10.0,
            xf=1000.0,
            tf=100.0
        )
        
        # Enable auto_step
        model.auto_step = True
        
        # Set up geometry
        model.set_geometry(
            bottom_width=10.0,
            side_slope=0.0,
            manning_coef=0.03,
            bed_slope=0.001
        )
        
        # Set initial conditions with high velocity to trigger adjustment
        model.set_initial_conditions(
            initial_depth=2.0,
            initial_velocity=4.0,  # High velocity
            initial_concentration=0.0
        )
        
        # Set output sections
        model.set_output_sections([0.0, 1000.0])
        
        # Set boundary conditions
        boundary_positions = {
            'Q': [0.0],
            'y': [1000.0]
        }
        
        time_array = np.array([0, 50, 100])
        
        q_values = np.array([[80.0, 80.0, 80.0]])  # m³/s (high flow)
        y_values = np.array([[2.0, 2.0, 2.0]])     # m
        
        boundary_data = {
            'Q': {'time': time_array, 'values': q_values},
            'y': {'time': time_array, 'values': y_values}
        }
        
        model.set_boundary_conditions(boundary_positions, boundary_data)
        
        # Set initial courant_max to trigger adjustment
        model.courant_max = 1.5
        
        # Run a single time step manually to test adjustment
        A1 = model._wet_area(model.b1, model.y1, model.m)
        B1 = model._top_width(model.b1, model.y1, model.m)
        
        # Calculate Courant number
        courant = model._courant_number(model.dt, model.dx, model.v1, model.gravity, A1, B1)
        
        # Original dt
        original_dt = model.dt
        
        # Run simulation with auto_step
        model.run(show_progress=False)
        
        # Check if dt was adjusted
        assert courant >= 1.0  # Should be unstable without adjustment
        assert model.dt < original_dt  # dt should be reduced
    
    def test_non_uniform_grid(self):
        """Test model with non-uniform initial conditions."""
        # Create a model with a non-uniform initial state
        model = SIHQUAL(
            dx=100.0,
            dt=10.0,
            xf=1000.0,
            tf=3600.0
        )
        
        # Set up geometry
        model.set_geometry(
            bottom_width=10.0,
            side_slope=0.0,
            manning_coef=0.03,
            bed_slope=0.001
        )
        
        # Create non-uniform initial depth (sloping water surface)
        y_initial = np.linspace(3.0, 1.0, model.dim)  # From 3m at upstream to 1m at downstream
        
        # Create non-uniform initial velocity
        v_initial = np.linspace(0.3, 0.7, model.dim)  # From 0.3 m/s at upstream to 0.7 m/s at downstream
        
        # Create non-uniform initial concentration
        c_initial = np.zeros(model.dim)
        c_initial[4:7] = 10.0  # Concentration pulse in the middle
        
        model.set_initial_conditions(y_initial, v_initial, c_initial)
        
        # Set output sections
        model.set_output_sections([0.0, 500.0, 1000.0])
        
        # Run a short simulation
        model.tf = 60.0  # 1 minute
        results = model.run(show_progress=False)
        
        # Check if results reflect the non-uniform initial conditions
        assert results.loc[0, (0, 'y')] > results.loc[0, (10, 'y')]  # Upstream depth > downstream depth
        assert results.loc[0, (0, 'Q')] != results.loc[0, (10, 'Q')]  # Flow should vary along the channel
        
        # Check if concentration pulse is captured
        assert results.loc[0, (5, 'c')] > 0  # Middle section should have concentration
    
    def test_friction_slope_calculation(self):
        """Test friction slope calculation."""
        # Test with different Manning's n values
        n_values = np.array([0.01, 0.03, 0.05])  # Low, medium, high roughness
        v = np.array([1.0, 1.0, 1.0])  # Same velocity
        rh = np.array([1.0, 1.0, 1.0])  # Same hydraulic radius
        
        sf = self.model._friction_slope(n_values, v, rh)
        
        # Higher n should result in higher friction slope
        assert sf[0] < sf[1] < sf[2]
        
        # Test with different velocities
        n = np.array([0.03, 0.03, 0.03])  # Same roughness
        v_values = np.array([0.5, 1.0, 2.0])  # Different velocities
        rh = np.array([1.0, 1.0, 1.0])  # Same hydraulic radius
        
        sf = self.model._friction_slope(n, v_values, rh)
        
        # Higher velocity should result in higher friction slope (quadratic relationship)
        assert sf[0] < sf[1] < sf[2]
        assert pytest.approx(sf[1] / sf[0]) == 4.0  # v^2 relationship
        assert pytest.approx(sf[2] / sf[1]) == 4.0  # v^2 relationship
    
    def test_x_to_index_conversion(self):
        """Test conversion from position to array index."""
        # Test with various positions
        positions = [0.0, 250.0, 500.0, 750.0, 1000.0]
        expected_indices = [0, 2, 5, 7, 10]  # For dx=100, xf=1000
        
        for pos, expected_idx in zip(positions, expected_indices):
            idx = self.model._x_to_index(pos)
            assert idx == expected_idx
    
    def test_avg_and_cdelta_functions(self):
        """Test average and central difference functions."""
        # Test vector
        test_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test average function
        avg_result = self.model._avg(test_vector)
        expected_avg = np.array([2.0, 3.0, 4.0])  # (1+3)/2, (2+4)/2, (3+5)/2
        assert np.array_equal(avg_result, expected_avg)
        
        # Test central difference function
        cdelta_result = self.model._cdelta(test_vector)
        expected_cdelta = np.array([2.0, 2.0, 2.0])  # 3-1, 4-2, 5-3
        assert np.array_equal(cdelta_result, expected_cdelta)
    
    def test_partial_boundary_conditions(self):
        """Test setting only some boundary conditions."""
        # Set only flow boundary condition
        boundary_positions = {
            'Q': [0.0]
        }
        
        time_array = np.array([0, 1800, 3600])
        q_values = np.array([[10.0, 20.0, 10.0]])
        
        boundary_data = {
            'Q': {'time': time_array, 'values': q_values}
        }
        
        self.model.set_boundary_conditions(boundary_positions, boundary_data)
        
        # Check if only flow boundary was set
        assert self.model.index_Q == [0]
        assert self.model.t_arr_Q is not None
        assert self.model.bQ_data is not None
        
        # Other boundaries should remain empty
        assert self.model.index_y == []
        assert self.model.index_c == []
    
    def test_multiple_boundary_points(self):
        """Test setting multiple boundary points for each variable."""
        # Set multiple boundary points
        boundary_positions = {
            'Q': [0.0, 500.0],  # Flow at upstream and middle
            'y': [1000.0],      # Water level at downstream
            'c': [0.0, 1000.0]  # Concentration at upstream and downstream
        }
        
        time_array = np.array([0, 1800, 3600])
        
        q_values = np.array([
            [10.0, 20.0, 10.0],  # Upstream flow
            [5.0, 10.0, 5.0]     # Middle flow
        ])
        
        y_values = np.array([[2.0, 2.5, 2.0]])  # Downstream water level
        
        c_values = np.array([
            [0.0, 5.0, 0.0],  # Upstream concentration
            [0.0, 2.0, 0.0]   # Downstream concentration
        ])
        
        boundary_data = {
            'Q': {'time': time_array, 'values': q_values},
            'y': {'time': time_array, 'values': y_values},
            'c': {'time': time_array, 'values': c_values}
        }
        
        self.model.set_boundary_conditions(boundary_positions, boundary_data)
        
        # Check if multiple boundary points were set correctly
        assert self.model.index_Q == [0, 5]
        assert self.model.index_y == [10]
        assert self.model.index_c == [0, 10]
        
        assert np.array_equal(self.model.bQ_data, q_values)
        assert np.array_equal(self.model.by_data, y_values)
        assert np.array_equal(self.model.bc_data, c_values)
    
    def test_zero_depth_handling(self):
        """Test handling of zero water depth."""
        # Create a model with zero initial depth in some sections
        model = SIHQUAL(
            dx=100.0,
            dt=10.0,
            xf=1000.0,
            tf=3600.0
        )
        
        # Set up geometry
        model.set_geometry(
            bottom_width=10.0,
            side_slope=0.0,
            manning_coef=0.03,
            bed_slope=0.001
        )
        
        # Create initial conditions with zero depth in some sections
        y_initial = np.ones(model.dim) * 2.0
        y_initial[3:6] = 0.0  # Zero depth in the middle
        
        v_initial = np.ones(model.dim) * 0.5
        c_initial = np.zeros(model.dim)
        
        model.set_initial_conditions(y_initial, v_initial, c_initial)
        model.set_output_sections([0.0, 500.0, 1000.0])
        
        # Set boundary conditions to ensure flow
        boundary_positions = {
            'Q': [0.0],
            'y': [1000.0]
        }
        
        time_array = np.array([0, 1800, 3600])
        q_values = np.array([[10.0, 10.0, 10.0]])
        y_values = np.array([[2.0, 2.0, 2.0]])
        
        boundary_data = {
            'Q': {'time': time_array, 'values': q_values},
            'y': {'time': time_array, 'values': y_values}
        }
        
        model.set_boundary_conditions(boundary_positions, boundary_data)
        
        # Run a short simulation
        model.tf = 60.0
        results = model.run(show_progress=False)
        
        # Check if simulation completed without errors
        assert isinstance(results, pd.DataFrame)
        assert not results.empty


if __name__ == "__main__":
    pytest.main(["-v", "test_sihqual.py"])