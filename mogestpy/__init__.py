"""Top-level package for MOGESTpy."""

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
        from .quantity.Hydrological import SMAP2
        return SMAP2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")