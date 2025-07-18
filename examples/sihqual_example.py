"""
Example usage of the SIHQUAL class for hydrodynamic and water quality simulation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import mogestpy
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mogestpy.quantity.Hydrodynamic.sihqual import SIHQUAL


def simple_channel_example():
    """
    Simple example of a rectangular channel with constant flow.
    """
    print("Running simple channel example...")
    
    # Basic parameters
    dx = 100.0  # m
    dt = 10.0   # s
    xf = 5000.0  # m (5 km)
    tf = 86400.0  # s (1 day)
    
    # Create model instance
    model = SIHQUAL(
        dx=dx,
        dt=dt,
        xf=xf,
        tf=tf,
        alpha=0.6,
        dispersion_coef=5.0
    )
    
    # Number of spatial points
    dim = int(xf / dx) + 1
    
    # Set up geometry (rectangular channel)
    b1 = np.ones(dim) * 20.0  # 20m wide channel
    m = np.ones(dim) * 0.0    # Rectangular channel
    n = np.ones(dim) * 0.035  # Manning's n
    So = np.ones(dim) * 0.0005  # Slope
    
    model.set_geometry(b1, m, n, So)
    
    # Set initial conditions
    y1 = np.ones(dim) * 1.5   # 1.5m depth
    v1 = np.ones(dim) * 0.5   # 0.5 m/s velocity
    c1 = np.zeros(dim)        # Zero initial concentration
    
    model.set_initial_conditions(y1, v1, c1)
    
    # Set reaction parameters (for BOD decay)
    Kd = np.ones(dim) * 0.0001  # Decay coefficient (1/s)
    Ks = np.zeros(dim)          # No source
    
    model.set_reaction_parameters(Kd, Ks)
    
    # Set output sections every 1 km
    output_positions = np.linspace(0, xf, 6)  # 0, 1km, 2km, 3km, 4km, 5km
    model.set_output_sections(output_positions)
    
    # Set boundary conditions
    # Upstream: flow and concentration
    # Downstream: water level
    boundary_positions = {
        'Q': [0.0],           # Upstream flow
        'y': [xf],            # Downstream water level
        'c': [0.0]            # Upstream concentration
    }
    
    # Time array for 24 hours with hourly values
    time_array = np.linspace(0, tf, 25)  # 0 to 24 hours
    
    # Flow with a simple diurnal pattern
    base_flow = 15.0  # m³/s
    flow_amplitude = 5.0  # m³/s
    q_values = np.array([
        base_flow + flow_amplitude * np.sin(2 * np.pi * t / 86400)
        for t in time_array
    ]).reshape(1, -1)
    
    # Constant downstream water level
    y_values = np.ones((1, 25)) * 1.5  # m
    
    # Concentration with a pulse at 6 hours
    c_values = np.zeros((1, 25))
    c_values[0, 6:12] = 10.0  # mg/L for 6 hours
    
    boundary_data = {
        'Q': {'time': time_array, 'values': q_values},
        'y': {'time': time_array, 'values': y_values},
        'c': {'time': time_array, 'values': c_values}
    }
    
    model.set_boundary_conditions(boundary_positions, boundary_data)
    
    # Add a lateral inflow at 2-2.5 km
    lateral_Q_segments = [(2000.0, 2500.0)]
    
    # Constant lateral inflow
    lateral_q_values = np.ones((1, 25)) * 0.002  # m³/s per meter
    
    lateral_Q_data = {
        'time': time_array,
        'values': lateral_q_values
    }
    
    # Lateral concentration
    lateral_c_segments = [(2000.0, 2500.0)]
    lateral_c_values = np.ones((1, 25)) * 5.0  # mg/L
    
    lateral_c_data = {
        'time': time_array,
        'values': lateral_c_values
    }
    
    model.set_lateral_inflows(
        lateral_Q_segments, lateral_Q_data,
        lateral_c_segments, lateral_c_data
    )
    
    # Run the simulation
    print("Running simulation...")
    results = model.run(show_progress=True)
    
    # Save results
    model.save_results(results, 'simple_channel_results.xlsx')
    print("Results saved to 'simple_channel_results.xlsx'")
    
    # Plot results
    plot_results(results)


def plot_results(results):
    """
    Plot simulation results.
    
    Args:
        results: DataFrame with simulation results
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Get unique sections
    sections = results.columns.get_level_values('section').unique()
    
    # Plot flow
    for section in sections:
        axes[0].plot(results.index, results[(section, 'Q')], 
                    label=f'Section {section}')
    
    axes[0].set_title('Flow')
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Flow (m³/s)')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot water depth
    for section in sections:
        axes[1].plot(results.index, results[(section, 'y')], 
                    label=f'Section {section}')
    
    axes[1].set_title('Water Depth')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Depth (m)')
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot concentration
    for section in sections:
        axes[2].plot(results.index, results[(section, 'c')], 
                    label=f'Section {section}')
    
    axes[2].set_title('Concentration')
    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylabel('Concentration (mg/L)')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('simple_channel_results.png')
    print("Plot saved to 'simple_channel_results.png'")
    plt.close()


def from_excel_example():
    """
    Example of loading a model from an Excel file.
    """
    print("\nRunning from_excel example...")
    
    # Check if the Excel file exists
    excel_file = 'SIHQUAL.xlsx'
    if not os.path.exists(excel_file):
        print(f"Excel file '{excel_file}' not found. Skipping this example.")
        return
    
    # Load model from Excel
    model = SIHQUAL.from_excel(excel_file)
    
    # Run the simulation
    print("Running simulation...")
    results = model.run(show_progress=True)
    
    # Save results
    model.save_results(results, 'excel_model_results.xlsx')
    print("Results saved to 'excel_model_results.xlsx'")


if __name__ == "__main__":
    # Run examples
    simple_channel_example()
    from_excel_example()