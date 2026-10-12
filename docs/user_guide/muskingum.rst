Muskingum routing
=================

The :mod:`mogestpy.quantity.hydrological.muskingum` module routes a flood
hydrograph through a river reach. The Muskingum method relates the storage in
the reach to a weighted mix of inflow :math:`I` and outflow :math:`O`:

.. math::

   S = K \left[ X I + (1 - X) O \right]

where :math:`K` is the storage coefficient, close to the travel time of the
reach, and :math:`X` (0 to 0.5) weights inflow and outflow. Smaller :math:`X`
gives more attenuation of the peak.

Linear Muskingum
----------------

Combined with the continuity equation, the linear method gives the outflow at
each time step :math:`\Delta t`:

.. math::

   O_{t+1} = C_0 I_{t+1} + C_1 I_t + C_2 O_t

.. math::

   C_0 = \frac{\Delta t - 2KX}{2K(1-X) + \Delta t}, \quad
   C_1 = \frac{\Delta t + 2KX}{2K(1-X) + \Delta t}, \quad
   C_2 = \frac{2K(1-X) - \Delta t}{2K(1-X) + \Delta t}

The coefficients sum to one, so the volume is conserved. They are
non-negative, which avoids oscillations, when
:math:`2KX \le \Delta t \le 2K(1 - X)`.

.. code-block:: python

   from mogestpy.quantity.hydrological.muskingum import Muskingum

   inflow = [10, 20, 40, 60, 80, 70, 50, 40, 30, 20, 15, 10]  # m³/s
   outflow = Muskingum.downstream_routing(inflow, k=24.0, x=0.3, dt=24.0)

``k`` and ``dt`` must be in the same unit (hours here). The outflow starts at
the first inflow value.

.. note::

   With a daily time step, reaches with travel time below half a day admit
   no :math:`X` that keeps the coefficients non-negative. In a basin network
   such reaches can use ``k = 0`` (pass-through); see :doc:`network`.

Non-linear Muskingum
--------------------

The non-linear form :math:`S = K \left[ X I + (1 - X) O \right]^m` is
integrated with a fourth-order Runge-Kutta scheme, downstream
(:meth:`~mogestpy.quantity.hydrological.muskingum.Muskingum.downstream_fork`,
from inflow to outflow) or upstream
(:meth:`~mogestpy.quantity.hydrological.muskingum.Muskingum.upstream_fork`,
from outflow to inflow):

.. code-block:: python

   outflow = Muskingum.downstream_fork(k=24.0, x=0.3, m=1.2, dt=24.0, inflow=inflow)

With ``m = 1`` the storage equation is the linear one, but the result differs
from ``downstream_routing``, which uses the discrete coefficients above
instead of integrating the storage equation.

.. seealso::

   :ref:`sphx_glr_auto_examples_plot_03_muskingum_routing.py`
