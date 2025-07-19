#!/usr/bin/env python3
"""
Example demonstrating SIHQUAL's support for any iterable input.

This example shows how you can use lists, tuples, numpy arrays, pandas Series,
ranges, and other iterables as input to SIHQUAL methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

# Try to import pandas, but don't fail if it's not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from mogestpy.quantity.Hydrodynamic.SIHQUAL import SIHQUAL

def example_with_lists():
    """Example using Python lists."""
    print("Example 1: Using Python lists")
    
    # Create model
    model = SIHQUAL(dx=100, dt=20, xf=1000, tf=3600)
    
    # Set geometry using lists
    bottom_widths = [8.0, 10.0, 12.0, 10.0, 8.0]  # Varying channel width
    side_slopes = [0.0, 0.0, 0.0, 0.0, 0.0]  # Rectangular channel
    manning_coefs = [0.025, 0.03, 0.035, 0.03, 0.025]  # Varying roughness
    bed_slopes = [0.001, 0.001, 0.001, 0.001, 0.001]  # Uniform slope
    
    model.set_geometry(
        bottom_width=bottom_widths,
        side_slope=side_slopes,
        manning_coef=manning_coefs,
        bed_slope=bed_slopes
    )
    
    # Set initial conditions using lists
    depths = [1.5, 2.0, 2.5, 2.0, 1.5]  # Varying initial depth
    velocities = [0.8, 0.6, 0.4, 0.6, 0.8]  # Varying initial velocity
    concentrations = [0.0] * 5  # Zero initial concentration
    
    model.set_initial_conditions(
        initial_depth=depths,
        initial_velocity=velocities,
        initial_concentration=concentrations
    )
    
    # Set reaction parameters
    model.set_uniform_reaction_parameters(decay_coef=0.0001)
    
    # Set output sections using a list
    model.set_output_sections([0, 250, 500, 750, 1000])
    
    # Set boundary conditions using lists
    times = [0, 1800, 3600]  # Time points
    flows = [8.0, 12.0, 8.0]  # Flow values
    levels = [1.8, 2.0, 1.8]  # Level values
    
    model.set_simple_upstream_flow(flow_values=flows, times=times)
    model.set_simple_downstream_level(level_values=levels, times=times)
    
    print("✓ Model created successfully using lists")
    return model

def example_with_tuples():
    """Example using Python tuples."""
    print("Example 2: Using Python tuples")
    
    # Create model
    model = SIHQUAL(dx=100, dt=20, xf=1000, tf=3600)
    
    # Set geometry using tuples
    bottom_widths = (10.0, 10.0, 10.0, 10.0, 10.0)  # Uniform width
    side_slopes = (0.5, 0.3, 0.0, 0.3, 0.5)  # Varying side slopes
    manning_coefs = (0.03,) * 5  # Uniform roughness using tuple multiplication
    bed_slopes = (0.002, 0.0015, 0.001, 0.0015, 0.002)  # Varying slope
    
    model.set_geometry(
        bottom_width=bottom_widths,
        side_slope=side_slopes,
        manning_coef=manning_coefs,
        bed_slope=bed_slopes
    )
    
    # Set initial conditions using tuples
    model.set_initial_conditions(
        initial_depth=(2.0, 2.0, 2.0, 2.0, 2.0),
        initial_velocity=(0.5, 0.5, 0.5, 0.5, 0.5),
        initial_concentration=(0.0, 0.0, 0.0, 0.0, 0.0)
    )
    
    model.set_uniform_reaction_parameters()
    model.set_output_sections((0, 500, 1000))  # Tuple for output sections
    
    # Simple boundary conditions
    model.set_simple_upstream_flow(flow_values=10.0)
    model.set_simple_downstream_level(level_values=2.0)
    
    print("✓ Model created successfully using tuples")
    return model

def example_with_numpy_arrays():
    """Example using NumPy arrays."""
    print("Example 3: Using NumPy arrays")
    
    # Create model
    model = SIHQUAL(dx=100, dt=20, xf=1000, tf=3600)
    
    # Create spatial coordinate
    x = np.linspace(0, 1000, 11)
    
    # Set geometry using numpy arrays with spatial variation
    bottom_widths = 8.0 + 4.0 * np.sin(x / 1000 * np.pi)  # Sinusoidal variation
    side_slopes = np.zeros_like(x)  # Rectangular channel
    manning_coefs = 0.025 + 0.01 * (x / 1000)  # Increasing roughness downstream
    bed_slopes = np.ones_like(x) * 0.001  # Uniform slope
    
    model.set_geometry(
        bottom_width=bottom_widths,
        side_slope=side_slopes,
        manning_coef=manning_coefs,
        bed_slope=bed_slopes
    )
    
    # Set initial conditions using numpy arrays
    depths = 1.5 + 0.5 * np.cos(x / 1000 * np.pi)  # Cosine variation
    velocities = np.ones_like(x) * 0.6  # Uniform velocity
    concentrations = np.zeros_like(x)  # Zero concentration
    
    model.set_initial_conditions(
        initial_depth=depths,
        initial_velocity=velocities,
        initial_concentration=concentrations
    )
    
    model.set_uniform_reaction_parameters()
    model.set_output_sections(np.array([0, 250, 500, 750, 1000]))
    
    # Boundary conditions with numpy arrays
    times = np.array([0, 1800, 3600])
    flows = np.array([9.0, 11.0, 9.0])
    
    model.set_simple_upstream_flow(flow_values=flows, times=times)
    model.set_simple_downstream_level(level_values=2.0)
    
    print("✓ Model created successfully using NumPy arrays")
    return model

def example_with_ranges():
    """Example using Python ranges and other iterables."""
    print("Example 4: Using ranges and other iterables")
    
    # Create model
    model = SIHQUAL(dx=100, dt=20, xf=1000, tf=3600)
    
    # Set uniform geometry (scalars)
    model.set_uniform_geometry(
        bottom_width=10.0,
        side_slope=0.0,
        manning_coef=0.03,
        bed_slope=0.001
    )
    
    model.set_uniform_initial_conditions(depth=2.0, velocity=0.5)
    model.set_uniform_reaction_parameters()
    
    # Set output sections using a range
    model.set_output_sections(range(0, 1001, 200))  # Every 200m
    
    model.set_simple_upstream_flow(flow_values=10.0)
    model.set_simple_downstream_level(level_values=2.0)
    
    print("✓ Model created successfully using ranges")
    return model

def example_with_pandas():
    """Example using pandas Series (if available)."""
    if not HAS_PANDAS:
        print("Example 5: Pandas not available, skipping")
        return None
        
    print("Example 5: Using pandas Series")
    
    # Create model
    model = SIHQUAL(dx=100, dt=20, xf=1000, tf=3600)
    
    # Create a DataFrame with channel properties
    x_positions = np.linspace(0, 1000, 11)
    channel_data = pd.DataFrame({
        'x': x_positions,
        'bottom_width': 8.0 + 2.0 * np.random.random(11),  # Random variation
        'side_slope': np.zeros(11),
        'manning': 0.025 + 0.01 * np.random.random(11),
        'bed_slope': np.ones(11) * 0.001
    })
    
    # Set geometry using pandas Series
    model.set_geometry(
        bottom_width=channel_data['bottom_width'],
        side_slope=channel_data['side_slope'],
        manning_coef=channel_data['manning'],
        bed_slope=channel_data['bed_slope']
    )
    
    # Initial conditions using pandas Series
    initial_data = pd.Series([2.0] * 11, name='depth')
    model.set_initial_conditions(
        initial_depth=initial_data,
        initial_velocity=0.5,
        initial_concentration=0.0
    )
    
    model.set_uniform_reaction_parameters()
    model.set_output_sections([0, 500, 1000])
    
    model.set_simple_upstream_flow(flow_values=10.0)
    model.set_simple_downstream_level(level_values=2.0)
    
    print("✓ Model created successfully using pandas Series")
    return model

def example_mixed_types():
    """Example mixing different iterable types."""
    print("Example 6: Mixing different iterable types")
    
    # Create model
    model = SIHQUAL(dx=100, dt=20, xf=1000, tf=3600)
    
    # Mix different types
    model.set_geometry(
        bottom_width=10.0,  # Scalar
        side_slope=[0.0, 0.1, 0.0, 0.1, 0.0],  # List
        manning_coef=(0.03, 0.03, 0.03, 0.03, 0.03),  # Tuple
        bed_slope=np.array([0.001, 0.001, 0.001, 0.001, 0.001])  # NumPy array
    )
    
    model.set_initial_conditions(
        initial_depth=2.0,  # Scalar
        initial_velocity=[0.5, 0.6, 0.5, 0.6, 0.5],  # List
        initial_concentration=np.zeros(5)  # NumPy array
    )
    
    model.set_uniform_reaction_parameters()
    model.set_output_sections(range(0, 1001, 250))  # Range
    
    # Boundary conditions with different types
    model.set_simple_upstream_flow(
        flow_values=[8.0, 12.0, 8.0],  # List
        times=(0, 1800, 3600)  # Tuple
    )
    model.set_simple_downstream_level(level_values=2.0)  # Scalar
    
    print("✓ Model created successfully mixing different iterable types")
    return model

def main():
    """Run all examples."""
    print("SIHQUAL Iterable Support Examples")
    print("=" * 40)
    
    examples = [
        example_with_lists,
        example_with_tuples,
        example_with_numpy_arrays,
        example_with_ranges,
        example_with_pandas,
        example_mixed_types
    ]
    
    models = []
    for example_func in examples:
        try:
            model = example_func()
            if model is not None:
                models.append(model)
            print()
        except Exception as e:
            print(f"❌ Error in {example_func.__name__}: {e}")
            print()
    
    print("=" * 40)
    print(f"✅ Successfully created {len(models)} models using different iterable types!")
    print("\nKey benefits:")
    print("• Use Python lists for simple, readable input")
    print("• Use tuples for immutable data")
    print("• Use NumPy arrays for mathematical operations")
    print("• Use pandas Series for data analysis workflows")
    print("• Use ranges for regular spacing")
    print("• Mix different types as needed")
    print("• Full backward compatibility with existing code")

if __name__ == "__main__":
    main()