"""Sphinx configuration for MOGESTpy."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "MOGESTpy"
author = "Dario Hachisu Hossoda"
copyright = "2026, Dario Hachisu Hossoda"


def _read_project_version() -> str:
    """Read the project version without requiring an installed package."""
    with (ROOT / "pyproject.toml").open("rb") as pyproject:
        data = tomllib.load(pyproject)

    return data["tool"]["poetry"]["version"]


try:
    release = _pkg_version("mogestpy")
except PackageNotFoundError:
    release = _read_project_version()

# Sphinx expects `version` (short X.Y) and `release` (full)
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "vitepress"
