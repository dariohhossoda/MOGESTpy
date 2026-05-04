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

The Sphinx documentation dependencies are optional. Install the same
requirements used by Read the Docs with:

.. code-block:: console

   pip install -r docs/requirements.txt

When using Poetry, install the documentation dependency group with:

.. code-block:: console

   poetry install --with docs

Then build the HTML documentation with:

.. code-block:: console

   sphinx-build -b html docs docs/_build/html

Read the Docs
-------------

Read the Docs uses the repository-level ``.readthedocs.yaml`` file. The build
configuration points to ``docs/conf.py`` and installs the packages listed in
``docs/requirements.txt`` before running Sphinx.
