"""
Daily rainfall-runoff with SMAP
===============================

Simulate the daily discharge of a basin with :class:`~mogestpy.quantity.hydrological.smap.SmapD`
and inspect the state of its reservoirs.
"""

# %%
# Forcing data
# ------------
# The model needs daily precipitation and potential evapotranspiration, in
# mm/day, with the same length and no missing values.

import matplotlib.pyplot as plt
import pandas as pd

from mogestpy.quantity.hydrological.smap import SmapD

forcing = pd.read_csv("data/biritiba.csv", index_col="day")
forcing.describe()

# %%
# Model setup
# -----------
# ``Ad`` is the drainage area (km²). The other arguments are the SMAP
# parameters (see :doc:`/user_guide/smap`); ``Tuin`` is the initial soil
# moisture as a fraction and ``Ebin`` the initial base flow (m³/s).

model = SmapD(
    Ad=70.2,
    Str=100.0,
    Crec=19.6,
    Capc=30.0,
    kkt=47.5,
    k2t=1.43,
    Ai=2.0,
    Tuin=0.05,
    Ebin=0.0,
)

# %%
# Run the simulation
# ------------------
# :meth:`~mogestpy.quantity.hydrological.smap.SmapD.run_to_list` returns the
# discharge (m³/s) of each day.

discharge = model.run_to_list(forcing["precipitation"], forcing["evapotranspiration"])

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
ax.plot(forcing.index, discharge, color="tab:blue", linewidth=0.8)
ax.set_xlabel("Day")
ax.set_ylabel("Discharge (m³/s)")
ax.set_xlim(forcing.index[0], forcing.index[-1])

rain = ax.twinx()
rain.bar(forcing.index, forcing["precipitation"], color="tab:gray", width=1)
rain.set_ylim(forcing["precipitation"].max() * 3, 0)
rain.set_ylabel("Precipitation (mm/day)")
plt.show()

# %%
# Internal states
# ---------------
# :meth:`~mogestpy.quantity.hydrological.smap.SmapD.run_to_dataframe` also
# returns the fluxes and the storage of each reservoir: soil (``Rsolo``),
# surface (``Rsup``) and groundwater (``Rsub``), in mm.

states = model.run_to_dataframe(forcing["precipitation"], forcing["evapotranspiration"])
states.head()

# %%

fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, layout="constrained")
for ax, column, label in zip(
    axes,
    ["Rsolo", "Rsup", "Rsub"],
    ["Soil", "Surface", "Groundwater"],
):
    ax.plot(states.index, states[column], linewidth=0.8)
    ax.set_ylabel(f"{label} (mm)")
axes[-1].set_xlabel("Day")
plt.show()
