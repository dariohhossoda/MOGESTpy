Contributing
============

Bug reports, feature requests and pull requests are welcome on
`GitHub <https://github.com/dariohhossoda/MOGESTpy>`_. The full guide, with
commit and code style, is in
`CONTRIBUTING.md <https://github.com/dariohhossoda/MOGESTpy/blob/main/CONTRIBUTING.md>`_.
In short:

1. Open an issue with one of the
   `templates <https://github.com/dariohhossoda/MOGESTpy/issues/new/choose>`_.
2. Create a branch for that issue from ``development``, named
   ``<issue number>-<short-description>``.
3. Add tests and documentation for the change.
4. Open a pull request to ``development`` filling in the template.

``main`` only receives release pull requests from ``development``.

Development environment
-----------------------

.. code-block:: console

   git clone https://github.com/dariohhossoda/MOGESTpy.git
   cd MOGESTpy
   poetry install --with dev,test
   poetry run pytest -m "not slow"
   poetry run black --check .

Building the documentation
--------------------------

.. code-block:: console

   poetry install --with docs
   cd docs
   poetry run sphinx-build -b html . _build/html

The examples in ``docs/examples`` run during the build, so a broken example
fails it.
