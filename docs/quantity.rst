Quantity Models
===============

The :mod:`mogestpy.quantity` package groups the flow-related models used by
MOGESTpy. It includes hydrological tools for rainfall-runoff and routing, as
well as hydrodynamic solvers for one-dimensional flow.

Hydrological Models
-------------------

The hydrological subpackage provides the SMAP rainfall-runoff model, the
Muskingum routing methods, the reservoir mass-balance helper and the network
routing workflow that combines SMAP with Muskingum.

Example
~~~~~~~

.. code-block:: python

   from mogestpy.quantity.hydrological.routing import run_network
   from mogestpy.quantity.hydrological.smap import SmapD

Hydrodynamic Models
-------------------

The hydrodynamic subpackage contains the Saint-Venant formulation and the
SIHQUAL solver for coupled hydrodynamic and water-quality simulations.

Example
~~~~~~~

.. code-block:: python

   from mogestpy.quantity.hydrodynamic.sihqual import SIHQUAL

   model = SIHQUAL(dx=100, dt=10, xf=1000, tf=3600)
