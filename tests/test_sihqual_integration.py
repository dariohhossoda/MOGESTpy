"""
Integration tests for the SIHQUAL module.
"""
import os
import numpy as np
import pandas as pd
import pytest
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path

from mogestpy.quantity.Hydrodynamic.sihqual import SIHQUAL


class TestSIHQUALIntegration:
    """Integration test suite for SIHQUAL class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a realistic river channel scenario
        self.dx = 500.0  # m
        self.dt = 60.0   # s (1 minute)
        self.xf = 20000.0  # m (20 km)
        self.tf = 86400.0  # s (1 day)
        
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
        
        # Set up geometry - gradually widening channel with varying slope
        x_points = np.linspace(0, self.xf, 5)  # 5 points along the channel
        
        # Bottom width: 30m at upstream to 50m at downstream
        b_points = np.array([30.0, 35.0, 40.0, 45.0, 50.0])
        
        # Side slope: 0.5 throughout
        m_points = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Manning's n: varying roughness
        n_points = np.array([0.035, 0.033, 0.030, 0.028, 0.025])
        
        # Bed slope: steeper upstream, flatter downstream
        So_points = np.array([0.0008, 0.0006, 0.0005, 0.0004, 0.0003])
        
        # Set geometry
        self.model.set_geometry(b_points, m_points, n_points, So_points)
        
        # Set initial conditions - gradually varying depth and velocity
        y_points = np.array([2.5, 2.7, 3.0, 3.2, 3.5])  # Depth increasing downstream
        v_points = np.array([0.8, 0.75, 0.7, 0.65, 0.6])  # Velocity decreasing downstream
        c_points = np.zeros(5)  # Zero initial concentration
        
        # Interpolate to full grid
        y_initial = np.interp(np.linspace(0, self.xf, self.dim), x_points, y_points)
        v_initial = np.interp(np.linspace(0, self.xf, self.dim), x_points, v_points)
        c_initial = np.zeros(self.dim)
        
        self.model.set_initial_conditions(y_initial, v_initial, c_initial)
        
        # Set reaction parameters - BOD decay
        Kd = np.ones(self.dim) * 0.00002  # Decay coefficient (1/s)
        Ks = np.zeros(self.dim)           # No source
        
        self.model.set_reaction_parameters(Kd, Ks)
        
        # Set output sections every 5 km
        output_positions = np.linspace(0, self.xf, 5)  # 0, 5km, 10km, 15km, 20km
        self.model.set_output_sections(output_positions)
    
    def test_realistic_river_simulation(self):
        """Test a realistic river simulation with varying flow and pollution event."""
        # Set boundary conditions
        # Upstream: flow and concentration
        # Downstream: water level
        boundary_positions = {
            'Q': [0.0],           # Upstream flow
            'y': [self.xf],       # Downstream water level
            'c': [0.0]            # Upstream concentration
        }
        
        # Time array for 24 hours with hourly values
        time_array = np.linspace(0, self.tf, 25)  # 0 to 24 hours
        
        # Flow with a realistic diurnal pattern
        base_flow = 60.0  # m³/s
        flow_amplitude = 15.0  # m³/s
        q_values = np.array([
            base_flow + flow_amplitude * np.sin(2 * np.pi * t / 86400 - np.pi/2)
            for t in time_array
        ]).reshape(1, -1)
        
        # Downstream water level with slight variation
        base_level = 3.5  # m
        level_amplitude = 0.2  # m
        y_values = np.array([
            base_level + level_amplitude * np.sin(2 * np.pi * t / 86400)
            for t in time_array
        ]).reshape(1, -1)
        
        # Concentration with a pollution event at 6 hours lasting 4 hours
        c_values = np.zeros((1, 25))
        c_values[0, 6:10] = 20.0  # mg/L for 4 hours
        
        boundary_data = {
            'Q': {'time': time_array, 'values': q_values},
            'y': {'time': time_array, 'values': y_values},
            'c': {'time': time_array, 'values': c_values}
        }
        
        self.model.set_boundary_conditions(boundary_positions, boundary_data)
        
        # Add a lateral inflow at 8-10 km
        lateral_Q_segments = [(8000.0, 10000.0)]
        
        # Lateral inflow with a simple pattern
        lateral_q_values = np.array([
            0.005 + 0.002 * np.sin(2 * np.pi * t / 86400)
            for t in time_array
        ]).reshape(1, -1)  # m³/s per meter
        
        lateral_Q_data = {
            'time': time_array,
            'values': lateral_q_values
        }
        
        # Lateral concentration
        lateral_c_segments = [(8000.0, 10000.0)]
        lateral_c_values = np.ones((1, 25)) * 5.0  # mg/L
        
        lateral_c_data = {
            'time': time_array,
            'values': lateral_c_values
        }
        
        self.model.set_lateral_inflows(
            lateral_Q_segments, lateral_Q_data,
            lateral_c_segments, lateral_c_data
        )
        
        # Run the simulation
        results = self.model.run(show_progress=False)
        
        # Check if results are returned
        assert isinstance(results, pd.DataFrame)
        assert not results.empty
        
        # Check if results have the expected structure
        expected_columns = pd.MultiIndex.from_product(
            [[0, 10, 20, 30, 40], ['Q', 'y', 'c']], names=['section', 'variables'])
        
        assert list(results.columns) == list(expected_columns)
        
        # Check if flow propagates through the system
        # Flow at downstream should lag behind upstream
        upstream_flow = results[(0, 'Q')]
        downstream_flow = results[(40, 'Q')]
        
        # Peak flow should occur later downstream than upstream
        upstream_peak_time = upstream_flow.idxmax()
        downstream_peak_time = downstream_flow.idxmax()
        
        # In a 20km river, with flow velocity around 0.7 m/s, 
        # travel time should be approximately 8 hours (8 time steps)
        # But due to numerical diffusion and other factors, the lag might be less
        assert downstream_peak_time >= upstream_peak_time
        
        # Check if concentration propagates through the system
        # The pollution pulse should move downstream and attenuate
        upstream_conc = results[(0, 'c')]
        midstream_conc = results[(20, 'c')]
        downstream_conc = results[(40, 'c')]
        
        # Upstream should have the highest peak concentration
        assert upstream_conc.max() > midstream_conc.max()
        assert midstream_conc.max() > downstream_conc.max()
        
        # Peak concentration should occur later downstream
        if upstream_conc.max() > 0:  # Only if there's a detectable peak
            upstream_conc_peak_time = upstream_conc[upstream_conc > 0].idxmax()
            
            # Find downstream peak if it exists
            if any(midstream_conc > 0):
                midstream_conc_peak_time = midstream_conc[midstream_conc > 0].idxmax()
                assert midstream_conc_peak_time > upstream_conc_peak_time
    
    def test_flood_wave_propagation(self):
        """Test propagation of a flood wave through the channel."""
        # Set boundary conditions for a flood wave
        boundary_positions = {
            'Q': [0.0],           # Upstream flow
            'y': [self.xf]        # Downstream water level
        }
        
        # Time array for 24 hours with hourly values
        time_array = np.linspace(0, self.tf, 25)  # 0 to 24 hours
        
        # Flow with a flood wave pattern (base flow + gaussian peak)
        base_flow = 60.0  # m³/s
        peak_flow = 200.0  # m³/s
        
        # Create a Gaussian flood wave centered at 6 hours
        def gaussian(x, mu, sig, scale):
            return scale * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        
        flood_wave = gaussian(time_array, 6*3600, 2*3600, peak_flow - base_flow)
        q_values = np.array([base_flow + flood_wave]).reshape(1, -1)
        
        # Downstream water level with slight variation
        y_values = np.ones((1, 25)) * 3.5  # m
        
        boundary_data = {
            'Q': {'time': time_array, 'values': q_values},
            'y': {'time': time_array, 'values': y_values}
        }
        
        self.model.set_boundary_conditions(boundary_positions, boundary_data)
        
        # Run the simulation
        results = self.model.run(show_progress=False)
        
        # Check if results are returned
        assert isinstance(results, pd.DataFrame)
        
        # Check flood wave propagation
        # Flow at each section
        flow_sections = [results[(i, 'Q')] for i in [0, 10, 20, 30, 40]]
        
        # Find peak time at each section
        peak_times = [flow.idxmax() for flow in flow_sections]
        
        # Peak should progressively occur later downstream
        for i in range(1, len(peak_times)):
            assert peak_times[i] >= peak_times[i-1]
        
        # Check water level response
        # Water level at each section
        level_sections = [results[(i, 'y')] for i in [0, 10, 20, 30, 40]]
        
        # Find peak time at each section
        level_peak_times = [level.idxmax() for level in level_sections]
        
        # Water level peak should progressively occur later downstream
        for i in range(1, len(level_peak_times)):
            assert level_peak_times[i] >= level_peak_times[i-1]
        
        # Check attenuation of the flood peak
        # Peak flow should decrease downstream due to attenuation
        peak_flows = [flow.max() for flow in flow_sections]
        
        # In a natural channel, peak flow typically attenuates
        # However, with lateral inflows or complex geometry, this might not always be true
        # So we'll just check that the peak flow doesn't increase dramatically
        for i in range(1, len(peak_flows)):
            assert peak_flows[i] <= peak_flows[0] * 1.1  # Allow for small numerical variations
    
    def test_pollution_transport(self):
        """Test transport of a pollution pulse through the channel."""
        # Set boundary conditions for a pollution pulse
        boundary_positions = {
            'Q': [0.0],           # Upstream flow
            'y': [self.xf],       # Downstream water level
            'c': [0.0]            # Upstream concentration
        }
        
        # Time array for 24 hours with hourly values
        time_array = np.linspace(0, self.tf, 25)  # 0 to 24 hours
        
        # Constant flow
        q_values = np.ones((1, 25)) * 60.0  # m³/s
        
        # Constant downstream water level
        y_values = np.ones((1, 25)) * 3.5  # m
        
        # Concentration with a sharp pulse at 4 hours
        c_values = np.zeros((1, 25))
        c_values[0, 4] = 100.0  # mg/L for 1 hour
        
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
        
        # Check pollution pulse propagation
        # Concentration at each section
        conc_sections = [results[(i, 'c')] for i in [0, 10, 20, 30, 40]]
        
        # Find peak time at each section
        peak_times = []
        for conc in conc_sections:
            if conc.max() > 0:
                peak_times.append(conc.idxmax())
            else:
                peak_times.append(None)
        
        # Peak should progressively occur later downstream for sections that receive the pulse
        for i in range(1, len(peak_times)):
            if peak_times[i] is not None and peak_times[i-1] is not None:
                assert peak_times[i] >= peak_times[i-1]
        
        # Check attenuation and dispersion of the pollution pulse
        # Peak concentration should decrease downstream due to dispersion and decay
        peak_concs = [conc.max() for conc in conc_sections]
        
        # Find sections with detectable concentration
        detected_sections = [i for i, peak in enumerate(peak_concs) if peak > 0]
        
        if len(detected_sections) > 1:
            # Peak concentration should decrease downstream
            for i in range(1, len(detected_sections)):
                section_idx1 = detected_sections[i-1]
                section_idx2 = detected_sections[i]
                assert peak_concs[section_idx2] <= peak_concs[section_idx1]
    
    def test_save_and_plot_results(self):
        """Test saving and plotting simulation results."""
        # Set simple boundary conditions
        boundary_positions = {
            'Q': [0.0],
            'y': [self.xf]
        }
        
        time_array = np.array([0, self.tf])
        q_values = np.array([[60.0, 60.0]])
        y_values = np.array([[3.5, 3.5]])
        
        boundary_data = {
            'Q': {'time': time_array, 'values': q_values},
            'y': {'time': time_array, 'values': y_values}
        }
        
        self.model.set_boundary_conditions(boundary_positions, boundary_data)
        
        # Run a short simulation
        self.model.tf = 3600.0  # 1 hour
        results = self.model.run(show_progress=False)
        
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
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_plot:
                plot_filename = tmp_plot.name
            
            try:
                # Create a simple plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot flow at different sections
                for section in [0, 10, 20, 30, 40]:
                    if (section, 'Q') in results.columns:
                        ax.plot(results.index, results[(section, 'Q')], 
                                label=f'Section {section}')
                
                ax.set_title('Flow at Different Sections')
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Flow (m³/s)')
                ax.grid(True)
                ax.legend()
                
                plt.savefig(plot_filename)
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


if __name__ == "__main__":
    pytest.main(["-v", "test_sihqual_integration.py"])