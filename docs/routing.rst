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
   Muskingum routing parameters. ``k`` must be non-negative and must be
   positive for basins that receive upstream flow. ``x`` must be between 0 and
   0.5.

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

Output Data
-----------

The result is a ``pandas.DataFrame`` indexed by the common time steps shared by
precipitation and evapotranspiration. Each column is a subcatchment ID, and
values are simulated discharge in cubic metres per second.

Validation
----------

The module validates required parameter columns, duplicated IDs, invalid
``downstream_id`` references, network cycles, missing forcing columns, null
forcing values and non-numeric forcing values before running the simulation.

Related Modules
---------------

* :mod:`mogestpy.quantity.hydrological.smap`
* :mod:`mogestpy.quantity.hydrological.muskingum`
