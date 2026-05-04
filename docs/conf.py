"""Sphinx configuration for MOGESTpy."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "MOGESTpy"
author = "Dario Hachisu Hossoda"
copyright = "2026, Dario Hachisu Hossoda"
# Try to obtain the package version from the installed package or source
try:
    # Prefer importing the package directly (works when source is on sys.path)
    import mogestpy

    release = getattr(mogestpy, "__version__", None) or mogestpy.version
except Exception:
    try:
        from importlib.metadata import version as _il_version

        release = _il_version("mogestpy")
    except Exception:
        # Last resort: hardcoded default
        release = "2.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
