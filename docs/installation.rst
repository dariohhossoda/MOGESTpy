Installation
============

Install the published package from PyPI with:

.. code-block:: console

   pip install mogestpy

For local development, clone the repository and install it from the repository
root:

.. code-block:: console

   pip install .

Editable installs are useful when changing the source code:

.. code-block:: console

   pip install -e .

Documentation Dependencies
--------------------------

The Sphinx documentation dependencies are optional. When using Poetry, install
them with:

.. code-block:: console

   poetry install --with docs

Then build the HTML documentation with:

.. code-block:: console

   sphinx-build -b html docs docs/_build/html
