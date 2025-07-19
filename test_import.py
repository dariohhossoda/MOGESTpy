#!/usr/bin/env python3
"""
Simple test script to verify SIHQUAL module import and basic functionality.
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    from mogestpy.quantity.Hydrodynamic.SIHQUAL import SIHQUAL
    print("✓ SIHQUAL import successful")
    
    # Test basic initialization
    model = SIHQUAL(dx=100.0, dt=10.0, xf=1000.0, tf=3600.0)
    print("✓ SIHQUAL initialization successful")
    
    # Test geometry setup
    model.set_geometry(
        bottom_width=10.0,
        side_slope=0.0,
        manning_coef=0.03,
        bed_slope=0.001
    )
    print("✓ Geometry setup successful")
    
    # Test initial conditions
    model.set_initial_conditions(
        initial_depth=2.0,
        initial_velocity=0.5,
        initial_concentration=0.0
    )
    print("✓ Initial conditions setup successful")
    
    # Test reaction parameters
    model.set_reaction_parameters(
        decay_coef=0.0001,
        source_coef=0.0
    )
    print("✓ Reaction parameters setup successful")
    
    # Test output sections
    model.set_output_sections([0.0, 500.0, 1000.0])
    print("✓ Output sections setup successful")
    
    print("All basic tests passed!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)