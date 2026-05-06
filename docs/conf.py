"""Sphinx configuration for MOGESTpy."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import sys
import mogestpy


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "MOGESTpy"
author = "Dario Hachisu Hossoda"
copyright = "2026, Dario Hachisu Hossoda"


release = mogestpy.__version__
version = release

autodoc_mock_imports = ["numpy"]


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
bibtex_bibfiles = ["references.bib"]