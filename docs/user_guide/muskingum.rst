Muskingum routing
=================

The :mod:`mogestpy.quantity.hydrological.muskingum` module routes a flood
hydrograph through a river reach, with :math:`K` and :math:`X` given by the
user (linear and non-linear Muskingum) or derived from the reach
characteristics (Muskingum-Cunge, :cite:`cunge69`). The Muskingum method relates the storage in
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
   such reaches can use ``k = 0`` (pass-through; see :doc:`network`) or be
   routed with `Muskingum-Cunge`_, which works on a shorter internal step.

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

Muskingum-Cunge
---------------

The :class:`~mogestpy.quantity.hydrological.muskingum.MuskingumCunge` class
follows the formulation of the MGB model :cite:`collischonn20`. For a reach
of length :math:`L`, slope :math:`S`, width :math:`B`, Manning's roughness
:math:`n` and reference discharge :math:`Q_0`, the kinematic celerity of a
wide rectangular channel is

.. math::

   c_0 = \frac{5}{3} \frac{Q_0^{0.4} S^{0.3}}{n^{0.6} B^{0.4}}

The reach is split into :math:`N` equal sub-reaches of length
:math:`\Delta x = L / N`, with :math:`N` given by the ideal sub-reach length
of :cite:`fread93`:

.. math::

   \Delta x_{ideal} = \frac{c_0 \Delta t}{2}
   \left( 1 + \sqrt{1 + \frac{1.5 \, Q_0}{B S \Delta t \, c_0^2}} \right)

Each sub-reach uses

.. math::

   K = \frac{\Delta x}{c_0}, \qquad
   X = \frac{1}{2} \left( 1 - \frac{Q_0}{B S c_0 \Delta x} \right)

with :math:`X` bounded to 0.2–0.5. The internal time step :math:`\Delta t`
(1 h by default) is reduced when needed to keep the routing coefficients
non-negative (:math:`2KX \le \Delta t \le 2K(1-X)`). The inflow is held
constant within each model time step, routed through the cascade of
sub-reaches on the internal step, and the outflow of each model step is the
mean of its sub-steps. Reaches with travel time shorter than a day are
therefore routed on a daily model, where the linear Muskingum would need
:math:`K = 0`.

The reference discharge can be regionalized from the drainage area
:math:`A` (km²) as :math:`Q_0 = A^{0.7} / 3`, as in MGB.

.. code-block:: python

   from mogestpy.quantity.hydrological.muskingum import MuskingumCunge

   area = 2500.0  # upstream drainage area (km²)
   outflow = MuskingumCunge.downstream_routing(
       inflow,  # daily discharge (m³/s)
       length=60_000.0,  # m
       q_ref=area**0.7 / 3,  # m³/s
       manning=0.035,
       width=1.5 * area**0.5,  # m
       slope=2e-4,  # m/m
   )

In a network simulation, ``MuskingumCunge.router()`` builds the ``router``
argument of :func:`~mogestpy.quantity.hydrological.routing.run_network`, which
reads ``length``, ``q_ref``, ``manning``, ``width`` and ``slope`` from each row
of the parameter table (see :doc:`network`).

.. seealso::

   :ref:`sphx_glr_auto_examples_plot_03_muskingum_routing.py`

References
----------

.. bibliography::
   :filter: docname in docnames
