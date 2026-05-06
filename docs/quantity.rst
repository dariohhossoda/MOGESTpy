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

SMAP
~~~~

The Soil Moisture Accounting Procedure (SMAP) is a rainfall-runoff model that
simulates the transformation of precipitation into runoff.



.. code-block:: python

   from mogestpy.quantity.hydrological.routing import run_network
   from mogestpy.quantity.hydrological.smap import SmapD

Hydrodynamic Models
-------------------

The implemented hydrodynamic model named SIHQUAL is a one-dimensional solver for the Saint-Venant equations.


.. math::
   :label: eq:sihqual

   \begin{cases}
   \frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = 0 \\
   \frac{\partial Q}{\partial t} + \frac{\partial }{\partial x} \left( \frac{Q^2}{A} + g I_1 \right) = g (I_2 - I_1) - g S_f
   \end{cases}

