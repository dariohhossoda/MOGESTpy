====
SMAP
====

The Soil Moisture Accounting Procedure (SMAP) is a rainfall-runoff model that
simulates the transformation of precipitation into runoff based on conceptual reservoirs representing different states. The formulation follows Lopes (1982).

The MOGESTpy implementation of SMAP includes two versions: a daily timestep version and a monthly timestep version. Both versions are designed to be flexible and can be calibrated to fit observed data for specific catchments.

Daily timestep
--------------

The Daily version of SMAP consists of 3 conceptual reservoirs: surface, soil and subsurface. This version can be accessed through the :mod:`mogestpy.quantity.hydrological.smap.SmapD` class.

.. figure:: _static/smapd.svg
   :alt: SMAP Daily Schematic
   :align: center

   Daily SMAP Schematic

The SmapD parameters are described in the table below.

.. list-table:: Daily SMAP parameters
    :header-rows: 1
    :widths: 18 46 22 14 14

    * - Parameter
      - Description
      - Range
      - Default
      - Unit
    * - Str
      - Soil saturation.
      - 100 to 2000
      - 100
      - mm
    * - Crec
      - Recession coefficient.
      - 0 to 20
      - 0
      - %
    * - Capc
      - Field capacity.
      - 30 to 50
      - 40
      - %
    * - kkt
      - Base flow recession coefficient.
      - 30 to 180
      - 30
      - d^-1
    * - k2t
      - Surface runoff recession coefficient.
      - 0.2 to 10
      - 0.2
      - d^-1
    * - Ad
      - Drainage area.
      - No explicit bound
      - 1.0
      - km^2
    * - Tuin
      - Initial soil moisture content.
      - 0 to 1
      - 0.0
      - Fraction
    * - Ebin
      - Initial base flow.
      - No explicit bound
      - 0.0
      - mm
    * - Ai
      - Initial abstraction.
      - 2 to 5
      - 2.5
      - mm


Examples
^^^^^^^^
Simulating
++++++++++

To use the daily version of SMAP, you can create an instance of the SmapD class and provide the necessary parameters. Here is an example of how to set up and run a simulation using the daily version of SMAP.

.. code-block:: python

   from mogestpy.quantity.hydrological.smap import SmapD

   model = SmapD(
    Str=100,
    Crec=19.6125,
    Capc=30,
    kkt=47.53,
    k2t=1.430,
    Ai=2,
    Tuin=.05,
    Ebin=0.1,
    Ad=70.2
    )

   precipitations = [10, 20, 15, 5, 0]  # Example precipitation data
   evapotranspirations = [2, 3, 1, 0.5, 0]  # Example evapotranspiration data

   discharges = model.run_to_list(precipitations, evapotranspirations)


Calibration example
^^^^^^^^^^^^^^^^^^^

The daily model can be calibrated against observed discharge data with the
``calibrate`` method.

.. code-block:: python

   from mogestpy.quantity.hydrological.smap import SmapD

   model = SmapD() # Initialize with default parameters

   result = model.calibrate(
       prec_arr=precipitations, # as list/np.array/pd.Series
       etp_arr=evapotranspirations, # as list/np.array/pd.Series
       eval_arr=observed_discharge, # as list/np.array/pd.Series
       variables=["Str", "Crec", "Capc", "kkt", "k2t", "Ai"],
       disp=True,
   )

   print(result.x)
   print(result.fun)

By default the calibration uses the Kling-Gupta Efficiency (KGE) as the objective function, but other metrics can be used by specifying the ``metric`` argument in the ``calibrate`` method.

Monthly timestep
----------------

The Monthly version of SMAP consists of 2 conceptual reservoirs: soil and subsurface and can be access through the :mod:`mogestpy.quantity.hydrological.smap.SmapM` class. This version is designed for applications where monthly data is more appropriate or available, and it simplifies the model structure while still capturing the essential hydrological processes.

.. figure:: _static/smapm.svg
   :alt: SMAP Monthly Schematic
   :name: smapm
   :align: center

   Monthly SMAP Schematic

The monthly SMAP parameters are described in the table below.

.. list-table:: Monthly SMAP parameters
    :header-rows: 1
    :widths: 18 46 22 14 14

    * - Parameter
      - Description
      - Range
      - Default
      - Unit
    * - Str
      - Soil saturation.
      - 400 to 5000
      - 1000
      - mm
    * - Pes
      - Exponent for soil evaporation.
      - 1 to 10
      - 1
      - Dimensionless
    * - Crec
      - Recharge coefficient.
      - 0 to 1
      - 0.5
      - Dimensionless
    * - kkt
      - Base flow recession coefficient.
      - 1 to 6
      - 1.5
      - months
    * - Tuin
      - Initial soil moisture content.
      - 0 to 1
      - 0.5
      - Fraction
    * - Ebin
      - Initial base flow.
      - No explicit bound
      - 0.1
      - m^3/s
    * - Ad
      - Drainage area.
      - No explicit bound
      - 1.0
      - km^2



Examples
^^^^^^^^

Simulating
++++++++++

Similarly, to use the monthly version of SMAP, you can create an instance of the SmapM class and provide the necessary parameters. Here is an example of how to set up and run a simulation using the monthly version of SMAP.

.. code-block:: python

   from mogestpy.quantity.hydrological.smap import SmapM

   model = SmapM(
    Str=1000,
    Pes=1,
    Crec=0.5,
    kkt=1.5,
    Tuin=0.5,
    Ebin=0.1,
    Ad=1,
    )

   precipitations = [100, 200, 150, 50, 0]  # Example precipitation data
   evapotranspirations = [20, 30, 10, 5, 0]  # Example evapotranspiration data

   discharges = model.run_to_list(precipitations, evapotranspirations)

Calibration example
^^^^^^^^^^^^^^^^^^^

The monthly model uses the same ``calibrate`` method.

.. code-block:: python

   from mogestpy.quantity.hydrological.smap import SmapM

   model = SmapM()

   result = model.calibrate(
       prec_arr=precipitations, # as list/np.array/pd.Series
       etp_arr=evapotranspirations, # as list/np.array/pd.Series
       eval_arr=observed_discharge, # as list/np.array/pd.Series
       variables=["Str", "Pes", "Crec", "kkt"],
       disp=True,
   )

   print(result.x)
   print(result.fun)


