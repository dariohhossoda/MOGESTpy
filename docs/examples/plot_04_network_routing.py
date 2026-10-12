"""
Basin network with SMAP and Muskingum
=====================================

Simulate three subcatchments connected in a network with
:func:`~mogestpy.quantity.hydrological.routing.run_network`: SMAP generates
the local discharge of each subcatchment and Muskingum routes the upstream
flow to the outlet.
"""

# %%
# Network and parameters
# ----------------------
# Two headwater subcatchments (``A`` and ``B``) drain into the outlet
# subcatchment ``C``. Each row of ``params`` holds the SMAP parameters of a
# subcatchment, the Muskingum ``k`` and ``x`` of the reach that crosses it
# (``k`` in days), and ``downstream_id``, empty at the outlet. ``tuin`` is a
# percentage here.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mogestpy.quantity.hydrological.routing import run_network

params = pd.DataFrame(
    {
        "id": ["A", "B", "C"],
        "area": [70.2, 45.0, 30.0],
        "str": [100.0, 150.0, 120.0],
        "crec": [19.6, 15.0, 10.0],
        "capc": [30.0, 35.0, 40.0],
        "kkt": [47.5, 60.0, 50.0],
        "k2t": [1.43, 2.0, 1.8],
        "ai": [2.0, 2.5, 2.5],
        "tuin": [5.0, 5.0, 5.0],
        "ebin": [0.0, 0.0, 0.0],
        "k": [0.0, 0.0, 1.5],
        "x": [0.2, 0.2, 0.2],
        "downstream_id": ["C", "C", np.nan],
    }
)
params

# %%
# Forcing
# -------
# One column per subcatchment, indexed by time. Here all subcatchments share
# the example series; a real study would use the areal mean over each one.

forcing = pd.read_csv("data/biritiba.csv", index_col="day")
ids = params["id"]
precipitation = pd.DataFrame({sid: forcing["precipitation"] for sid in ids})
evapotranspiration = pd.DataFrame({sid: forcing["evapotranspiration"] for sid in ids})

# %%
# Simulation
# ----------
# ``return_natural_flow=True`` also returns the local SMAP discharge of each
# subcatchment, before the upstream flow is added.

total, natural = run_network(
    params, precipitation, evapotranspiration, return_natural_flow=True
)

window = slice(1000, 1180)
fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
ax.stackplot(
    total.index[window],
    natural["A"].iloc[window],
    natural["B"].iloc[window],
    natural["C"].iloc[window],
    labels=["Local A", "Local B", "Local C"],
    alpha=0.5,
)
ax.plot(total.index[window], total["C"].iloc[window], color="black", label="Total at C")
ax.set_xlabel("Day")
ax.set_ylabel("Discharge (m³/s)")
ax.legend(loc="upper right")
plt.show()

# %%
# The outlet total differs from the sum of local flows only by the routing
# of the upstream flow through the reach of ``C``; volumes are conserved.

to_hm3 = 86400 / 1e6  # daily discharge (m³/s) to volume (hm³)
print(f"Local volumes, all subcatchments: {natural.sum().sum() * to_hm3:.1f} hm³")
print(f"Volume at the outlet:             {total['C'].sum() * to_hm3:.1f} hm³")

# %%
# Other options
# -------------
# * ``k = 0`` passes the upstream flow through, for reaches with travel time
#   negligible compared with the time step.
# * ``external_inflows`` adds known inflows from basins outside ``params``,
#   e.g. to calibrate a network group by group.
# * ``router`` replaces Muskingum with another routing function.
#
# See :doc:`/user_guide/network`.
