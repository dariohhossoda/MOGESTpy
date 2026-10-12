"""
Reservoir water quality with the zero-dimensional model
=======================================================

Simulate the concentration of a pollutant in a completely mixed reservoir
with :class:`~mogestpy.quality.zero_d.ZeroDimensional`.
"""

# %%
# Inputs
# ------
# Every input is a series with one value per time step, in consistent units:
# volume (m³), inflow and outflow (m³/s), inflow concentration (mg/L), contact
# area for settling (m²), first-order decay coefficient ``k`` (1/s) and
# settling velocity ``v`` (m/s). The time step is in seconds.

import matplotlib.pyplot as plt
import numpy as np

from mogestpy.quality.zero_d import ZeroDimensional

n_steps = 240
dt = 3600.0  # 1 h
time_h = np.arange(n_steps) * dt / 3600

volume = np.full(n_steps, 5.0e5)
inflow = np.full(n_steps, 10.0)
outflow = np.full(n_steps, 10.0)
area = np.full(n_steps, 1.0e5)

# Pollutant load entering during the first 48 hours
concentration_in = np.where(time_h < 48, 50.0, 0.0)

# %%
# Decay and settling
# ------------------
# Compare a conservative substance with one that decays and settles.

cases = {
    "Conservative": (np.zeros(n_steps), np.zeros(n_steps)),
    "Decay + settling": (np.full(n_steps, 1.0e-5), np.full(n_steps, 1.0e-6)),
}

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
ax.plot(time_h, concentration_in, color="black", linestyle="--", label="Inflow")
for label, (k, v) in cases.items():
    reactor = ZeroDimensional(volume, inflow, outflow, concentration_in, area, k, v, dt)
    reactor.RunModel()
    ax.plot(time_h, reactor.Output.ConcentrationOut, label=label)
ax.set_xlabel("Time (h)")
ax.set_ylabel("Concentration (mg/L)")
ax.set_xlim(time_h[0], time_h[-1])
ax.legend()
plt.show()

# %%
# With a residence time of ``V / Q`` ≈ 14 h, the reservoir approaches the
# inflow concentration within the loading period and is flushed afterwards;
# decay and settling lower the whole curve.
