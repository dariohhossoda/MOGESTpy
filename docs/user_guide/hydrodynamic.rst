Hydrodynamic models
===================

The :mod:`mogestpy.quantity.hydrodynamic` package solves one-dimensional
unsteady flow in open channels.

SIHQUAL
-------

:class:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL` solves the
Saint-Venant equations for a channel with trapezoidal cross-sections, coupled
with the advection-dispersion of a constituent with first-order decay:

.. math::
   :label: eq:sihqual

   \begin{cases}
   \frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = 0 \\
   \frac{\partial Q}{\partial t} + \frac{\partial }{\partial x} \left( \frac{Q^2}{A} + g I_1 \right) = g (I_2 - I_1) - g S_f
   \end{cases}

where :math:`A` is the wetted area, :math:`Q` the discharge, :math:`I_1` and
:math:`I_2` the pressure terms of the cross-section and :math:`S_f` the
friction slope from Manning's equation.

A simulation is set up in steps, all in SI units (m, s, m³/s, mg/L):

1. **Domain:** ``SIHQUAL(dx, dt, xf, tf)`` with the spatial step, time step,
   channel length and simulation time.
2. **Geometry:** :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.set_uniform_geometry`
   (bottom width, side slope, Manning's coefficient, bed slope) or
   :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.set_geometry` with
   one value per section.
3. **Initial conditions:** :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.set_uniform_initial_conditions`
   (depth, velocity, concentration) or
   :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.set_initial_conditions`.
4. **Boundary conditions:** :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.set_simple_upstream_flow`,
   :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.set_simple_downstream_level`
   or, for flow, level and concentration at any position,
   :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.set_boundary_conditions`.
5. **Optional terms:** lateral inflows and loads
   (:meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.set_lateral_inflows`)
   and decay and source coefficients
   (:meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.set_uniform_reaction_parameters`).
6. **Outputs:** :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.set_output_sections`
   or :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.set_evenly_spaced_output_sections`.
7. **Run:** :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.run` returns
   a DataFrame with discharge, depth and concentration at the output sections,
   one row per simulated day;
   :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.save_results` and
   :meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.plot_results`
   export and plot it.

:meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.create_simple_channel`
does steps 1–3 and 6 for a uniform rectangular channel, and
:meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.from_excel` reads a
whole setup from a spreadsheet.

The explicit scheme requires a Courant number below one; ``run`` stops with
an error otherwise, unless
:meth:`~mogestpy.quantity.hydrodynamic.sihqual.SIHQUAL.enable_auto_step_adjustment`
is on.

Saint-Venant building blocks
----------------------------

:mod:`mogestpy.quantity.hydrodynamic.saint_venant` provides
:class:`~mogestpy.quantity.hydrodynamic.saint_venant.TrapezoidalCrossSection`
(top width, wetted area and perimeter, normal depth) and
:class:`~mogestpy.quantity.hydrodynamic.saint_venant.SaintVenant` (Courant and
Froude numbers, friction slope) for custom solvers.
