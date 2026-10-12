Quantity Models
===============

The :mod:`mogestpy.quantity` package groups the flow-related models used by
MOGESTpy. It includes hydrological tools for rainfall-runoff and routing, as
well as hydrodynamic solvers for one-dimensional flow.



Hydrological Routing
--------------------

The :mod:`mogestpy.quantity.hydrological.muskingum` module provides the linear
and non-linear Muskingum methods, with storage coefficient :math:`K` and
weighting factor :math:`X` given by the user, and the Muskingum-Cunge method,
which derives both from the reach characteristics :cite:`cunge69`.

Muskingum-Cunge
^^^^^^^^^^^^^^^

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
of the parameter table (see :doc:`routing`).

Hydrodynamic Models
-------------------

The implemented hydrodynamic model named SIHQUAL is a one-dimensional solver for the Saint-Venant equations.


.. math::
   :label: eq:sihqual

   \begin{cases}
   \frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = 0 \\
   \frac{\partial Q}{\partial t} + \frac{\partial }{\partial x} \left( \frac{Q^2}{A} + g I_1 \right) = g (I_2 - I_1) - g S_f
   \end{cases}

References
----------

.. bibliography::
   :filter: docname in docnames
