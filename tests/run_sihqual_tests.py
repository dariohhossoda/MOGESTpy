"""
Test runner for SIHQUAL tests.
"""
import os
import sys
import pytest

# Add parent directory to path to import mogestpy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_tests():
    """Run all SIHQUAL tests."""
    print("Running SIHQUAL tests...")
    
    # Run unit tests
    print("\n=== Running unit tests ===")
    pytest.main(["-v", "test_sihqual.py"])
    
    # Run integration tests
    print("\n=== Running integration tests ===")
    pytest.main(["-v", "test_sihqual_integration.py"])
    
    # Run Excel loading tests
    print("\n=== Running Excel loading tests ===")
    pytest.main(["-v", "test_sihqual_excel.py"])
    
    # Run deprecated module tests
    print("\n=== Running deprecated module tests ===")
    pytest.main(["-v", "test_sihqual_deprecated.py"])


if __name__ == "__main__":
    run_tests()