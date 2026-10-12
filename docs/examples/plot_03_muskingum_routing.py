"""
Flood routing with Muskingum
============================

Route a flood hydrograph through a river reach with the linear Muskingum
method, see how ``K`` and ``X`` change the outflow, and derive them from the
reach characteristics with Muskingum-Cunge.
"""

# %%
# Inflow hydrograph
# -----------------
# Hourly inflow at the upstream end of the reach (m³/s).

import matplotlib.pyplot as plt
import numpy as np

from mogestpy.quantity.hydrological.muskingum import Muskingum, MuskingumCunge

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

# %%
# Muskingum-Cunge
# ---------------
# :class:`~mogestpy.quantity.hydrological.muskingum.MuskingumCunge` computes
# ``K`` and ``X`` from the length, slope, width and roughness of the reach and a
# reference discharge, and routes on an internal sub-step. Here the time step
# of the hydrograph is one hour (3600 s), and the reach is 30 km long.

reach = {"length": 30_000.0, "q_ref": 60.0, "manning": 0.035, "width": 30.0}

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
ax.plot(hours, inflow, color="black", label="Inflow")
for slope in [1e-3, 1e-4]:
    outflow = MuskingumCunge.downstream_routing(
        inflow, slope=slope, dt=3600.0, sub_dt=600.0, **reach
    )
    params = MuskingumCunge.parameters(slope=slope, dt=600.0, **reach)
    ax.plot(
        hours,
        outflow,
        label=f"S = {slope:g}: {params['n_subreaches']} sub-reaches, "
        f"K = {params['k'] / 3600:.2f} h and X = {params['x']:.2f} per sub-reach",
    )
ax.set_xlabel("Time (h)")
ax.set_ylabel("Discharge (m³/s)")
ax.set_xlim(hours[0], hours[-1])
ax.legend()
plt.show()

# %%
# A milder slope lowers the celerity, so the flood takes longer to cross the
# reach, and increases the physical diffusion, so the peak is attenuated more.
