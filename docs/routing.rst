Hydrological Network Routing
============================

The module :mod:`mogestpy.quantity.hydrological.routing` combines local
rainfall-runoff estimates from SMAP with upstream routing through the
Muskingum method. It is imported lazily, so users can load it only when a
network simulation is required.

.. code-block:: python

   from mogestpy.quantity.hydrological.routing import run_network

Input Data
----------

The function ``run_network`` expects three ``pandas.DataFrame`` objects:

``params``
   One row per subcatchment. Required columns are ``id``, ``area``, ``str``,
   ``crec``, ``capc``, ``kkt``, ``k2t``, ``ai``, ``tuin``, ``ebin``, ``k``,
   ``x`` and ``downstream_id``.

``precipitation``
   Time series with one column per subcatchment ID. Values are precipitation
   depths in millimetres per time step.

``evapotranspiration``
   Time series with one column per subcatchment ID. Values are potential
   evapotranspiration depths in millimetres per time step.

Parameter Table Format
----------------------

``id``
   Unique subcatchment identifier. The value must also exist as a column in
   the precipitation and evapotranspiration DataFrames.

``area``
   Drainage area in square kilometres.

``str``, ``crec``, ``capc``, ``kkt``, ``k2t``, ``ai``, ``tuin`` and ``ebin``
   SMAP parameters. ``tuin`` must be informed as a percentage from 0 to 100;
   the routing module converts it to the fraction expected by ``SmapD``.

``k`` and ``x``
   Muskingum routing parameters, with ``k`` in time-step units. ``k`` must be
   non-negative and ``x`` must be between 0 and 0.5. ``k = 0`` passes the
   upstream flow through unchanged. This suits reaches whose travel time is
   negligible compared with the time step: with a daily step, a reach with
   ``k`` below half a day admits no ``x`` that keeps the Muskingum
   coefficients non-negative.

``downstream_id``
   Identifier of the directly downstream subcatchment. Use a null value for an
   outlet basin.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mogestpy.quantity.hydrological.routing import run_network

   params = pd.DataFrame(
       {
           "id": ["headwater", "outlet"],
           "area": [513.37, 400.0],
           "str": [1559.91, 1200.0],
           "crec": [0.25, 0.2],
           "capc": [40.0, 40.0],
           "kkt": [83.63, 70.0],
           "k2t": [6.49, 5.5],
           "ai": [2.5, 2.5],
           "tuin": [50.0, 45.0],
           "ebin": [11.05, 8.0],
           "k": [1.0, 1.0],
           "x": [0.2, 0.2],
           "downstream_id": ["outlet", np.nan],
       }
   )

   index = pd.date_range("2026-01-01", periods=3, freq="D")
   precipitation = pd.DataFrame(
       {
           "headwater": [0.0, 5.0, 0.0],
           "outlet": [0.0, 3.0, 1.0],
       },
       index=index,
   )
   evapotranspiration = pd.DataFrame(
       {
           "headwater": [4.5, 4.5, 4.5],
           "outlet": [4.5, 4.5, 4.5],
       },
       index=index,
   )

   discharge = run_network(params, precipitation, evapotranspiration)

Optional Arguments
------------------

``external_inflows``
   ``pandas.DataFrame`` with the discharge (m3/s) entering subcatchments from
   basins that are not in ``params``. Columns are the IDs of the receiving
   subcatchments, and the index must cover all simulation time steps. Each
   column is added to the upstream flow of its subcatchment and routed with its
   ``k`` and ``x``. This allows simulating part of a network, for instance when
   calibrating groups of subcatchments from upstream to downstream:

   .. code-block:: python

      full = run_network(params, precipitation, evapotranspiration)
      outlet_only = run_network(
          params[params["id"] == "outlet"],
          precipitation,
          evapotranspiration,
          external_inflows=full[["headwater"]].rename(
              columns={"headwater": "outlet"}
          ),
      )  # same outlet discharge as ``full``

``router``
   Function that replaces Muskingum as the reach routing method, called as
   ``router(q_upstream, row)``. ``q_upstream`` is the summed upstream discharge
   (``numpy.ndarray``, m3/s) and ``row`` is the subcatchment row of ``params``,
   including any extra column, so method-specific reach parameters can be kept
   in the parameter table. The function must return the routed discharge with
   the same length.

   ``MuskingumCunge.router()`` routes each reach with the Muskingum-Cunge
   method, reading ``length``, ``q_ref``, ``manning``, ``width`` and ``slope``
   from the parameter table:

   .. code-block:: python

      from mogestpy.quantity.hydrological.muskingum import MuskingumCunge

      params["length"] = [12_000.0, 0.0]  # m; 0 = pass-through
      params["q_ref"] = params["area"] ** 0.7 / 3  # m3/s
      params["manning"] = 0.035
      params["width"] = 1.5 * params["area"] ** 0.5  # m
      params["slope"] = 1e-3  # m/m

      discharge = run_network(
          params,
          precipitation,
          evapotranspiration,
          router=MuskingumCunge.router(),
      )

``return_natural_flow``
   If ``True``, the function returns ``(total, natural)``, where ``natural`` is
   the local SMAP discharge of each subcatchment before adding upstream flow.

Output Data
-----------

The result is a ``pandas.DataFrame`` indexed by the common time steps shared by
precipitation and evapotranspiration. Each column is a subcatchment ID, and
values are simulated discharge in cubic metres per second. The discharge of a
subcatchment is its local SMAP discharge plus the routed sum of the discharges
of its upstream subcatchments:

.. math::

   Q_i = Q^{SMAP}_i + \mathrm{Muskingum}\left(\sum_{m \in U(i)} Q_m,\ k_i,\ x_i\right)

where :math:`U(i)` is the set of subcatchments directly upstream of
:math:`i`.

Validation
----------

The module validates required parameter columns, duplicated IDs, invalid
``downstream_id`` references, network cycles, missing forcing columns, null
forcing values and non-numeric forcing values before running the simulation.
External inflows must refer to existing IDs, cover all time steps and contain
only finite values. A custom router must return finite values with the same
length as its input.

Related Modules
---------------

* :mod:`mogestpy.quantity.hydrological.smap`
* :mod:`mogestpy.quantity.hydrological.muskingum`
