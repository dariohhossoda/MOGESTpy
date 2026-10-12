Water quality models
====================

The :mod:`mogestpy.quality` package has a completely mixed reservoir model and
a build-up/washoff model for pollutants on urban surfaces.

Zero-dimensional reservoir
--------------------------

:class:`~mogestpy.quality.zero_d.ZeroDimensional` treats the reservoir as a
continuously stirred tank (CSTR) :cite:`becker23`. The mass balance of a
constituent with concentration :math:`C` in a volume :math:`V` is

.. math::

   \frac{d(VC)}{dt} = Q_{in} C_{in} - Q_{out} C - k V C - v A_s C

with inflow :math:`Q_{in}` at concentration :math:`C_{in}`, outflow
:math:`Q_{out}`, first-order decay coefficient :math:`k` and settling velocity
:math:`v` over the contact area :math:`A_s`. It is solved with an implicit
Euler step, so it is stable for any time step. The concentration starts at
zero.

Every input is a sequence with one value per time step, in consistent units
(e.g. m³, m³/s, mg/L, m², 1/s, m/s and a time step in seconds):

.. code-block:: python

   from mogestpy.quality.zero_d import ZeroDimensional

   n = 48
   reactor = ZeroDimensional(
       volume=[5.0e5] * n,  # m³
       Qin=[10.0] * n,  # m³/s
       Qout=[10.0] * n,  # m³/s
       Cin=[50.0] * n,  # mg/L
       As=[1.0e5] * n,  # m²
       k=[1.0e-5] * n,  # 1/s
       v=[1.0e-6] * n,  # m/s
       timestep=3600.0,  # s
   )
   reactor.RunModel()

   concentration = reactor.Output.ConcentrationOut  # mg/L, one value per step

.. seealso::

   :ref:`sphx_glr_auto_examples_plot_05_zero_dimensional.py`

Build-up and washoff
--------------------

:class:`~mogestpy.quality.buwo.BuildUpWashoff` follows the SWMM approach for
one land use. While the surface flow stays below ``threshold_flow``, pollutant
mass builds up on the surface, limited to ``Bmax`` per unit area; above it,
the flow washes the mass off.

* Build-up (``BuMethod``): ``2`` exponential,
  :math:`B = B_{max}(1 - e^{-K_b t})`; ``3`` saturation,
  :math:`B = B_{max} t / (K_b + t)`.
* Washoff (``WoMethod``): ``1`` exponential, :math:`W = K_w q^{N_w} B`;
  ``2`` rating curve, :math:`W = K_w (1000\,q A)^{N_w}`; ``3`` event mean
  concentration, :math:`W = K_w q A`.

.. code-block:: python

   import numpy as np

   from mogestpy.quality.buwo import BuildUpWashoff

   surface_flow = np.array([0, 0, 0, 0, 5.0, 12.0, 3.0, 0, 0, 0, 0, 8.0, 0, 0])

   model = BuildUpWashoff(
       Bmax=50.0,
       Nb=1.0,
       Kb=0.3,
       threshold_flow=1.0,
       Nw=1.2,
       Kw=0.05,
       BuMethod=2,
       WoMethod=1,
       timestep_h=24.0,
       initial_buildup=100.0,
       area=10.0,
       area_fraction=0.4,
       surface_flow=surface_flow,
       landuse_name="urban",
   )
   model.Process()

   model.BuildUp  # mass on the surface at each step
   model.Washoff  # washed-off mass at each step

References
----------

.. bibliography::
   :filter: docname in docnames
