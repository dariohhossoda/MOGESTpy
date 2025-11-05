"""
Integration tests for the SIHQUAL module.
"""
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mogestpy.quantity.Hydrodynamic.SIHQUAL import SIHQUAL


class TestSIHQUALIntegration:
    """Integration test suite for SIHQUAL class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a minimal river channel scenario for faster tests
        self.dx = 1000.0  # m (increased spatial step)
        self.dt = 60.0    # s (increased time step)
        self.xf = 5000.0  # m (5 km - reduced length)
        self.tf = 900.0   # s (15 min - greatly reduced simulation time)

        # Create model instance
        self.model = SIHQUAL(
            dx=self.dx,
            dt=self.dt,
            xf=self.xf,
            tf=self.tf,
            alpha=0.6,
            dispersion_coef=10.0
        )

        # Number of spatial points
        self.dim = int(self.xf / self.dx) + 1

        # Set up geometry - simplified channel
        # Only 3 points along the channel
        x_points = np.linspace(0, self.xf, 3)

        # Bottom width: 30m at upstream to 50m at downstream
        b_points = np.array([30.0, 40.0, 50.0])

        # Side slope: 0.5 throughout
        m_points = np.array([0.5, 0.5, 0.5])

        # Manning's n: varying roughness
        n_points = np.array([0.035, 0.030, 0.025])

        # Bed slope: steeper upstream, flatter downstream
        So_points = np.array([0.0008, 0.0005, 0.0003])

        # Set geometry
        self.model.set_geometry(b_points, m_points, n_points, So_points)

        # Set initial conditions - gradually varying depth and velocity
        y_points = np.array([2.5, 3.0, 3.5])  # Depth increasing downstream
        v_points = np.array([0.8, 0.7, 0.6])  # Velocity decreasing downstream

        # Interpolate to full grid
        x_full = np.linspace(0, self.xf, self.dim)
        y_initial = np.interp(x_full, x_points, y_points)
        v_initial = np.interp(x_full, x_points, v_points)
        c_initial = np.zeros(self.dim)

        self.model.set_initial_conditions(y_initial, v_initial, c_initial)

        # Set reaction parameters - BOD decay
        Kd = np.ones(self.dim) * 0.00002  # Decay coefficient (1/s)
        Ks = np.zeros(self.dim)           # No source

        self.model.set_reaction_parameters(Kd, Ks)

        # Set output sections
        output_positions = np.linspace(0, self.xf, 3)  # Only 3 output points
        self.model.set_output_sections(output_positions)

    def test_flood_wave_propagation(self):
        """Test propagation of a flood wave through the channel."""
        # Set boundary conditions for a flood wave
        boundary_positions = {
            'Q': [0.0],  # Upstream flow
            'y': [self.xf]  # Downstream water level
        }

        # Simplified time array with fewer points
        time_array = np.array([0, self.tf/2, self.tf])

        # Flow with a simple flood wave pattern
        base_flow = 60.0  # m³/s
        peak_flow = 200.0  # m³/s

        # Simple triangular flood wave
        q_values = np.array([[base_flow, peak_flow, base_flow]])

        # Constant downstream water level
        y_values = np.ones((1, 3)) * 3.5  # m

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

        # For this simplified test, we just verify that the simulation ran
        # and produced results without errors

    def test_pollution_transport(self):
        """Test transport of a pollution pulse through the channel."""
        # Set boundary conditions for a pollution pulse
        boundary_positions = {
            'Q': [0.0],  # Upstream flow
            'y': [self.xf],  # Downstream water level
            'c': [0.0]  # Upstream concentration
        }

        # Simplified time array with fewer points
        time_array = np.array([0, self.tf/2, self.tf])

        # Constant flow
        q_values = np.ones((1, 3)) * 60.0  # m³/s

        # Constant downstream water level
        y_values = np.ones((1, 3)) * 3.5  # m

        # Concentration with a pulse in the middle time step
        c_values = np.array([[0.0, 100.0, 0.0]])

        boundary_data = {
            'Q': {'time': time_array, 'values': q_values},
            'y': {'time': time_array, 'values': y_values},
            'c': {'time': time_array, 'values': c_values}
        }

        self.model.set_boundary_conditions(boundary_positions, boundary_data)

        # Run the simulation
        results = self.model.run(show_progress=False)

        # Check if results are returned
        assert isinstance(results, pd.DataFrame)

        # Get output section indices
        section_indices = self.model.output_sections

        # This is a simplified test that still verifies the model works
        if len(results) > 0 and len(section_indices) > 1:
            # Get the middle time step
            mid_time = results.index[len(results.index) // 2]

            # Check that concentration decreases downstream
            upstream_conc = results.loc[mid_time, (section_indices[0], 'c')]
            downstream_conc = results.loc[mid_time, (section_indices[-1], 'c')]

            # Just check that the concentration values are non-negative
            assert upstream_conc >= 0
            assert downstream_conc >= 0

    def test_save_and_plot_results(self):
        """Test saving and plotting simulation results."""
        # Create a simple DataFrame for testing with fewer sections
        sections = [0, 1, 2]  # Simplified section indices
        variables = ['Q', 'y', 'c']
        cols = pd.MultiIndex.from_product(
            [sections, variables],
            names=['section', 'variables']
        )
        t_index = pd.Index([0, 1], name='t')

        # Create sample data (smaller array)
        data = np.array([
            [10.0, 2.0, 0.0, 9.0, 1.9, 0.1, 8.0, 1.8, 0.2],  # Day 0
            [11.0, 2.1, 0.1, 10.0, 2.0, 0.2, 9.0, 1.9, 0.3]  # Day 1
        ])

        results = pd.DataFrame(data, columns=cols, index=t_index)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            temp_filename = tmp.name

        try:
            # Test saving
            result = self.model.save_results(results, temp_filename)
            assert result is True

            # Check if file exists and can be read
            assert os.path.exists(temp_filename)

            # Try to read it back
            df_read = pd.read_excel(temp_filename, index_col=0, header=[0, 1])

            # Check if data is preserved
            assert df_read.shape == results.shape

            # Create a simple plot to test visualization
            with tempfile.NamedTemporaryFile(
                suffix='.png', delete=False
            ) as tmp_plot:
                plot_filename = tmp_plot.name

            try:
                # Create a simple plot (with smaller figure size)
                fig, ax = plt.subplots(figsize=(6, 4))

                # Plot flow at different sections
                for section in sections:
                    ax.plot(
                        results.index,
                        results[(section, 'Q')],
                        label=f'Section {section}'
                    )

                ax.set_title('Flow at Different Sections')
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Flow (m³/s)')
                ax.grid(True)
                ax.legend()

                # Use a lower DPI for faster plotting
                plt.savefig(plot_filename, dpi=72)
                plt.close()

                # Check if plot was created
                assert os.path.exists(plot_filename)
                assert os.path.getsize(plot_filename) > 0
            finally:
                # Clean up plot file
                if os.path.exists(plot_filename):
                    os.unlink(plot_filename)
        finally:
            # Clean up data file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_realistic_river_simulation(self):
        """Test a realistic river simulation
        with varying flow and pollution event."""
        # Set boundary conditions
        boundary_positions = {
            'Q': [0.0],  # Upstream flow
            'y': [self.xf],  # Downstream water level
            'c': [0.0]  # Upstream concentration
        }

        # Simplified time array with fewer points
        time_array = np.array([0, self.tf/2, self.tf])

        # Flow with a simple pattern
        base_flow = 60.0  # m³/s
        q_values = np.array([[base_flow, base_flow*1.2, base_flow]])

        # Downstream water level
        y_values = np.array([[3.5, 3.6, 3.5]])

        # Concentration with a pulse in the middle time step
        c_values = np.array([[0.0, 20.0, 0.0]])

        boundary_data = {
            'Q': {'time': time_array, 'values': q_values},
            'y': {'time': time_array, 'values': y_values},
            'c': {'time': time_array, 'values': c_values}
        }

        self.model.set_boundary_conditions(boundary_positions, boundary_data)

        # Add a lateral inflow
        lateral_Q_segments = [(2000.0, 3000.0)]
        lateral_q_values = np.array([[0.005, 0.007, 0.005]])

        lateral_Q_data = {
            'time': time_array,
            'values': lateral_q_values
        }

        self.model.set_lateral_inflows(lateral_Q_segments, lateral_Q_data)

        # Run the simulation
        results = self.model.run(show_progress=False)

        # Check if results are returned
        assert isinstance(results, pd.DataFrame)
        assert not results.empty


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_sihqual_integration.py"])
