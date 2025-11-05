"""
Example usage of the improved SIHQUAL class for hydrodynamic and water quality simulation.
"""
from mogestpy.quantity.Hydrodynamic.SIHQUAL import SIHQUAL
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import mogestpy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def simple_channel_example():
    """
    Simple example of a rectangular channel with constant flow using the improved API.
    """
    print("Running simple channel example with improved API...")

    # Create a simple channel model with default parameters
    model = SIHQUAL.create_simple_channel(
        length=5000,  # 5 km
        simulation_time=86400,  # 1 day
        bottom_width=20.0,
        manning=0.035,
        slope=0.0005
    )

    # Set simple boundary conditions
    model.set_simple_upstream_flow(flow_values=[15.0, 20.0, 15.0],
                                   # Flow varying over time
                                   times=[0, 43200, 86400])

    model.set_simple_downstream_level(
        level_values=1.5)  # Constant downstream level

    # Run the simulation
    print("Running simulation...")
    results = model.run(show_progress=True)

    # Save results
    model.save_results(results, 'simple_channel_results.xlsx')
    print("Results saved to 'simple_channel_results.xlsx'")

    # Plot results
    fig, ax = model.plot_results(results, variable='Q')
    plt.savefig('flow_results.png')
    plt.close()

    fig, ax = model.plot_results(results, variable='y')
    plt.savefig('depth_results.png')
    plt.close()

    print("Plots saved to 'flow_results.png' and 'depth_results.png'")


def trapezoidal_channel_example():
    """
    Example of a trapezoidal channel with pollution pulse.
    """
    print("\nRunning trapezoidal channel example...")

    # Create model
    model = SIHQUAL(dx=100.0, dt=10.0, xf=5000.0, tf=86400.0)

    # Set up geometry (trapezoidal channel)
    model.set_uniform_geometry(
        bottom_width=15.0,
        side_slope=1.5,  # 1.5:1 side slope (horizontal:vertical)
        manning_coef=0.03,
        bed_slope=0.0008
    )

    # Set initial conditions
    model.set_uniform_initial_conditions(
        depth=2.5,
        velocity=0.6,
        concentration=0.0
    )

    # Set reaction parameters
    model.set_uniform_reaction_parameters(
        decay_coef=0.00005,  # Decay coefficient
        source_coef=0.0      # No source
    )

    # Set output sections every 1 km
    model.set_evenly_spaced_output_sections(6)  # 0, 1km, 2km, 3km, 4km, 5km

    # Set boundary conditions
    # Time array for 24 hours with hourly values
    time_array = np.linspace(0, 86400, 25)  # 0 to 24 hours

    # Flow with a simple diurnal pattern
    base_flow = 25.0  # m³/s
    flow_values = base_flow + 5.0 * np.sin(2 * np.pi * time_array / 86400)

    # Concentration with a pulse at 6 hours
    conc_values = np.zeros_like(time_array)
    conc_values[6:12] = 10.0  # mg/L for 6 hours

    # Set boundary conditions
    model.set_simple_upstream_flow(flow_values=flow_values, times=time_array)
    model.set_simple_downstream_level(level_values=2.0)

    # Set upstream concentration boundary
    boundary_positions = {'c': [0.0]}
    boundary_data = {'c': {'time': time_array,
                           'values': np.array([conc_values])}}
    model.set_boundary_conditions(boundary_positions, boundary_data)

    # Run the simulation
    print("Running simulation...")
    results = model.run(show_progress=True)

    # Save results
    model.save_results(results, 'trapezoidal_channel_results.xlsx')
    print("Results saved to 'trapezoidal_channel_results.xlsx'")

    # Plot concentration results
    fig, ax = model.plot_results(results, variable='c')
    plt.savefig('concentration_results.png')
    plt.close()

    print("Plot saved to 'concentration_results.png'")


if __name__ == "__main__":
    # Run examples
    simple_channel_example()
    trapezoidal_channel_example()
