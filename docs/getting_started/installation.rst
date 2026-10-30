Installation
============

MOGESTpy requires Python 3.11 or later.

From PyPI
---------

Stable releases are published on `PyPI`_:

.. code-block:: console

   pip install mogestpy

The package depends on pandas, scikit-learn, SPOTPY and tqdm (which bring
NumPy and SciPy), installed automatically by pip. The examples in this documentation also use
Matplotlib for plots:

.. code-block:: console

   pip install matplotlib

PSO calibration (``pso_calibrate``) needs the optional ``pyswarms`` package,
installed with the ``pso`` extra:

.. code-block:: console

   pip install "mogestpy[pso]"

From source
-----------

To get unreleased changes or to contribute, clone the `GitHub repository`_
and install it in editable mode, so changes to the source take effect without
reinstalling:

.. code-block:: console

   git clone https://github.com/dariohhossoda/MOGESTpy.git
   cd MOGESTpy
   pip install -e .

Development uses Poetry; see the :doc:`contributing guide </about/contributing>`.

Check the installation
----------------------

.. code-block:: python

   import mogestpy

   print(mogestpy.__version__)

.. _GitHub repository: https://github.com/dariohhossoda/MOGESTpy
.. _PyPI: https://pypi.org/project/mogestpy/
