"""Top-level package for MOGESTpy."""

from importlib.metadata import PackageNotFoundError, version

__author__ = "Dario Hachisu Hossoda"
__email__ = "dario.hossoda@usp.br"

try:
    __version__ = version("mogestpy")
except PackageNotFoundError:
    # Fallback for local, non-installed usage
    __version__ = "0+unknown"

from . import quality, quantity
from .quantity.Hydrological import SMAP2

__all__ = ("quality", "quantity", "SMAP2", "__version__")