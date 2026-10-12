Quickstart
==========

This page runs the daily SMAP rainfall-runoff model, calibrates it and
routes its discharge through a river reach. Every block can be pasted into a
Python session in order.

Simulate a basin
----------------

:class:`~mogestpy.quantity.hydrological.smap.SmapD` turns daily precipitation
and potential evapotranspiration (mm/day) into discharge (m³/s) at the basin
outlet. ``Ad`` is the drainage area in km².

.. code-block:: python

   import numpy as np

   from mogestpy.quantity.hydrological.smap import SmapD

   rng = np.random.default_rng(0)
   days = 730
   precipitation = rng.gamma(shape=0.4, scale=12.0, size=days)  # mm/day
   evapotranspiration = 3.5 + np.sin(np.arange(days) * 2 * np.pi / 365)  # mm/day

   model = SmapD(Ad=70.2, Str=400.0, Crec=10.0, kkt=60.0, k2t=2.0, Tuin=0.3)
   discharge = model.run_to_list(precipitation, evapotranspiration)

Calibrate it
------------

Given an observed discharge series of the same length,
:meth:`~mogestpy.quantity.hydrological.smap.SmapD.fast_calibrate` fits the
chosen parameters by maximizing the Kling-Gupta Efficiency (KGE). Here the
"observed" series is the simulation above with noise, and calibration starts
from the default parameters:

.. code-block:: python

   observed = np.asarray(discharge) * rng.lognormal(sigma=0.1, size=days)

   new_model = SmapD(Ad=70.2, Tuin=0.3)
   result = new_model.fast_calibrate(
       precipitation,
       evapotranspiration,
       observed,
       variables=["Str", "Crec", "kkt", "k2t"],
   )

   print(dict(zip(["Str", "Crec", "kkt", "k2t"], result.x)))
   print("KGE:", -result.fun)

The calibrated values are set on ``new_model``, which can be run right away.

Route the discharge
-------------------

:meth:`~mogestpy.quantity.hydrological.muskingum.Muskingum.downstream_routing`
propagates the discharge through a reach with the linear Muskingum method.
``k`` and ``dt`` share the same unit (days here):

.. code-block:: python

   from mogestpy.quantity.hydrological.muskingum import Muskingum

   simulated = new_model.run_to_list(precipitation, evapotranspiration)
   routed = Muskingum.downstream_routing(simulated, k=1.5, x=0.2, dt=1.0)

Next steps
----------

* The :doc:`/user_guide/index` explains each model, its parameters and options.
* The :doc:`/auto_examples/index` have complete scripts with real forcing data
  and plots.
* :doc:`/user_guide/network` combines SMAP and routing over many
  subcatchments.
