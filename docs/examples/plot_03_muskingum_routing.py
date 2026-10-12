"""
Flood routing with Muskingum
============================

Route a flood hydrograph through a river reach with the linear Muskingum
method and see how ``K`` and ``X`` change the outflow.
"""

# %%
# Inflow hydrograph
# -----------------
# Hourly inflow at the upstream end of the reach (m³/s).

import matplotlib.pyplot as plt
import numpy as np

from mogestpy.quantity.hydrological.muskingum import Muskingum

hours = np.arange(72)
inflow = 20 + 180 * (hours / 12) ** 2 * np.exp(2 * (1 - hours / 12))

# %%
# Routing
# -------
# :meth:`~mogestpy.quantity.hydrological.muskingum.Muskingum.downstream_routing`
# takes ``K`` and the time step in the same unit (hours here). ``K`` is the
# travel time of the reach and ``X`` (0 to 0.5) weights inflow and outflow in
# the storage: smaller ``X`` means more attenuation. The coefficients stay
# non-negative when ``2KX <= dt <= 2K(1 - X)``.

dt = 1.0
cases = [(4.0, 0.2), (8.0, 0.2), (8.0, 0.4)]

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
ax.plot(hours, inflow, color="black", label="Inflow")
for k, x in cases:
    outflow = Muskingum.downstream_routing(inflow, k, x, dt)
    ax.plot(hours, outflow, label=f"K = {k:g} h, X = {x:g}")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Discharge (m³/s)")
ax.set_xlim(hours[0], hours[-1])
ax.legend()
plt.show()

# %%
# The routed hydrographs keep the inflow volume: the peak is delayed and
# attenuated.

for k, x in cases:
    outflow = np.asarray(Muskingum.downstream_routing(inflow, k, x, dt))
    print(
        f"K = {k:g} h, X = {x:g}: peak {outflow.max():6.1f} m³/s "
        f"at {hours[outflow.argmax()]} h, volume ratio {outflow.sum() / inflow.sum():.3f}"
    )
