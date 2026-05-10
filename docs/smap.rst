SMAP
----

The Soil Moisture Accounting Procedure (SMAP) is a rainfall-runoff model that
simulates the transformation of precipitation into runoff based on conceptual reservoirs representing different states :cite:`Lopes82`.

The MOGESTpy implementation of SMAP includes two versions: a daily timestep version and a monthly timestep version. Both versions are designed to be flexible and can be calibrated to fit observed data for specific catchments.

Daily timestep
~~~~~~~~~~~~~~

Daily version of SMAP

Parameters
^^^^^^^^^^

.. list-table:: Daily SMAP parameters
    :header-rows: 1
    :widths: 18 46 22 14

    * - Parameter
      - Description
      - Range
      - Unit
    * - Str
      - Soil saturation.
      - 100 to 2000
      - mm
    * - Crec
      - Recession coefficient.
      - 0 to 20
      - %
    * - Capc
      - Field capacity.
      - 30 to 50
      - %
    * - kkt
      - Base flow recession coefficient.
      - 30 to 180
      - d^-1
    * - k2t
      - Surface runoff recession coefficient.
      - 0.2 to 10
      - d^-1
    * - Ad
      - Drainage area.
      - No explicit bound
      - km^2
    * - Tuin
      - Initial soil moisture content.
      - 0 to 1
      - Fraction
    * - Ebin
      - Initial base flow.
      - No explicit bound
      - mm
    * - Ai
      - Initial abstraction.
      - 2 to 5
      - mm

.. image:: _static/smapd.svg
   :alt: SMAP Daily Schematic


Examples
^^^^^^^^

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


Monthly timestep
~~~~~~~~~~~~~~~~

Monthly version of SMAP

Parameters
^^^^^^^^^^

.. list-table:: Monthly SMAP parameters
    :header-rows: 1
    :widths: 18 46 22 14

    * - Parameter
      - Description
      - Range
      - Unit
    * - Str
      - Soil saturation.
      - 400 to 5000
      - mm
    * - Pes
      - Exponent for soil evaporation.
      - 1 to 10
      - Dimensionless
    * - Crec
      - Recharge coefficient.
      - 0 to 1
      - Dimensionless
    * - kkt
      - Base flow recession coefficient.
      - 1 to 6
      - months
    * - Tuin
      - Initial soil moisture content.
      - 0 to 1
      - Fraction
    * - Ebin
      - Initial base flow.
      - No explicit bound
      - m^3/s
    * - Ad
      - Drainage area.
      - No explicit bound
      - km^2

.. image:: _static/smapm.svg
   :alt: SMAP Monthly Schematic


Examples
^^^^^^^^

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



.. bibliography:: references.bib