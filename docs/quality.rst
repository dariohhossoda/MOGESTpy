Quality Models
==============

The :mod:`mogestpy.quality` package groups the water-quality models shipped
with MOGESTpy.

Zero-Dimensional Model
----------------------

The zero-dimensional model represents a well-mixed reservoir and is useful
for simple concentration balance studies.

Example
~~~~~~~

.. code-block:: python

   from mogestpy.quality import zero_d

   reactor = zero_d.ZeroDimensional(
       volume=[1000, 1000],
       Qin=[50, 50],
       Qout=[50, 50],
       Cin=[60, 60],
       As=[50, 50],
       k=[0.05, 0.05],
       v=[0.5, 0.5],
       timestep=50,
   )

Build-Up/Washoff Model
----------------------

The build-up/washoff model estimates pollutant accumulation on a surface and
its transport during runoff events.

Example
~~~~~~~

.. code-block:: python

   from mogestpy.quality import buwo
