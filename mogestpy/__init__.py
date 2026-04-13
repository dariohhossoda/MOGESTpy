"""
MOGESTpy
========

A water resources modeling library for Python.

The package is divided into two main submodules:
- `quantity`: Contains classes and functions related to water quantity modeling, such as the SMAP model.
- `quality`: Contains classes and functions related to water quality modeling.

Use the import as the examples below to access the different components of the library:
  >>> from mogestpy.quantity.Hydrological import SMAP2

Check the project repository for examples and documentation on how to use the library effectively at https://github.com/dariohhossoda/MOGESTpy
"""

from importlib.metadata import PackageNotFoundError, version

__author__ = "Dario Hachisu Hossoda"
__email__ = "dario.hossoda@usp.br"
_DIST_NAME = "MOGESTpy"

try:
    __version__ = version(_DIST_NAME)
except PackageNotFoundError:
    # Fallback for local, non-installed usage
    __version__ = "0+unknown"

from . import quality, quantity

__all__ = ("quality", "quantity", "SMAP2", "__version__")


def __getattr__(name):
    """Lazily import heavy/optional objects on first access.

    This avoids importing optional dependencies (e.g., SMAP2 and its
    transitive requirements) at mogestpy package import time.
    """
    if name == "SMAP2":
        # Import SMAP2 only when it is actually requested.
        from .quantity.Hydrological import smap

        return smap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
