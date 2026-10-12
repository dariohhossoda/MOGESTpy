:html_theme.sidebar_secondary.remove:

MOGESTpy
========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.12215704.svg
   :target: https://doi.org/10.5281/zenodo.12215704
   :alt: DOI

.. image:: https://img.shields.io/pypi/v/mogestpy
   :target: https://pypi.org/project/mogestpy/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://github.com/dariohhossoda/MOGESTpy/blob/main/LICENSE.md
   :alt: MIT license

**MOGESTpy** is a Python library of hydrological, hydrodynamic and water
quality models for water resources planning and management: SMAP
rainfall-runoff, Muskingum and Muskingum-Cunge routing, basin networks, the
SIHQUAL hydrodynamic and transport model, and reservoir and urban runoff
quality models.

.. code-block:: console

   pip install mogestpy

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :fas:`rocket` Getting started
      :link: getting_started/index
      :link-type: doc

      Install the package and run a first simulation and calibration in a
      few minutes.

   .. grid-item-card:: :fas:`book-open` User guide
      :link: user_guide/index
      :link-type: doc

      What each model represents, its equations, parameters and how to use
      it.

   .. grid-item-card:: :fas:`chart-line` Examples
      :link: auto_examples/index
      :link-type: doc

      Complete scripts with data, outputs and figures, downloadable as
      Python scripts or notebooks.

   .. grid-item-card:: :fas:`code` API reference
      :link: api/index
      :link-type: doc

      Every class, method and argument, generated from the source code.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   user_guide/index
   auto_examples/index
   api/index
   about/index
