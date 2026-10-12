"""Sphinx configuration for MOGESTpy."""

from pathlib import Path
import sys

# Document the source tree, not an installed release
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mogestpy  # noqa: E402

project = "MOGESTpy"
author = "Dario Hachisu Hossoda"
copyright = "2026, Dario Hachisu Hossoda"

release = mogestpy.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "examples"]

bibtex_bibfiles = ["references.bib"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "download_all_examples": False,
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
}

html_theme = "pydata_sphinx_theme"
html_title = f"MOGESTpy {release}"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/dariohhossoda/MOGESTpy",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/mogestpy/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Zenodo",
            "url": "https://doi.org/10.5281/zenodo.12215704",
            "icon": "fa-solid fa-quote-right",
        },
    ],
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}
html_context = {"default_mode": "auto"}
html_sidebars = {"index": [], "about/**": []}
