"""
Calibrating SMAP
================

Calibrate :class:`~mogestpy.quantity.hydrological.smap.SmapD` against observed
discharge and check the result on a separate validation period.
"""

# %%
# Observed discharge
# ------------------
# The example dataset has no discharge record, so the "observed" series is
# synthetic: SMAP with known parameters plus multiplicative noise. A real study
# would read the gauged discharge instead.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spotpy.objectivefunctions import kge

from mogestpy.quantity.hydrological.smap import SmapD

forcing = pd.read_csv("data/biritiba.csv", index_col="day")
precipitation = forcing["precipitation"]
evapotranspiration = forcing["evapotranspiration"]

reference = SmapD(
    Ad=70.2, Str=100.0, Crec=19.6, Capc=30.0, kkt=47.5, k2t=1.43, Ai=2.0, Tuin=0.05
)
rng = np.random.default_rng(42)
observed = np.asarray(reference.run_to_list(precipitation, evapotranspiration))
observed = observed * rng.lognormal(sigma=0.15, size=observed.size)

# %%
# Calibration and validation periods
# ----------------------------------
# The first 1500 days are used to calibrate and the remaining days to validate.

calibration = slice(0, 1500)
validation = slice(1500, None)

# %%
# Calibrate
# ---------
# Start from the default parameters. :meth:`~mogestpy.quantity.hydrological.smap.SmapD.fast_calibrate`
# runs Differential Evolution within the bounds of
# :meth:`~mogestpy.quantity.hydrological.smap.SmapD.bounds`, maximizing the
# Kling-Gupta Efficiency (KGE), and sets the calibrated values on the model.

model = SmapD(Ad=70.2, Tuin=0.05)

result = model.fast_calibrate(
    precipitation.iloc[calibration],
    evapotranspiration.iloc[calibration],
    observed[calibration],
    variables=["Str", "Crec", "kkt", "k2t"],
)

pd.DataFrame(
    {
        "calibrated": result.x,
        "reference": [reference.Str, reference.Crec, reference.kkt, reference.k2t],
    },
    index=["Str", "Crec", "kkt", "k2t"],
)

# %%
# Evaluate
# --------
# Run the calibrated model over the whole series and compute the KGE in each
# period. The objective function is minimized, so ``result.fun`` is ``-KGE``.

simulated = np.asarray(model.run_to_list(precipitation, evapotranspiration))

for name, period in [("calibration", calibration), ("validation", validation)]:
    print(f"KGE {name}: {kge(observed[period], simulated[period]):.3f}")

# %%

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
ax.plot(forcing.index, observed, color="black", linewidth=0.6, label="Observed")
ax.plot(forcing.index, simulated, color="tab:red", linewidth=0.8, label="Simulated")
ax.axvspan(1500, forcing.index[-1], color="tab:gray", alpha=0.15, label="Validation")
ax.set_xlabel("Day")
ax.set_ylabel("Discharge (m³/s)")
ax.set_xlim(forcing.index[0], forcing.index[-1])
ax.legend()
plt.show()

# %%
# Other options
# -------------
# * :meth:`~mogestpy.quantity.hydrological.smap.SmapD.calibrate` searches
#   continuous values instead of 4 decimal places and does not set them on the
#   model.
# * ``obj_func`` takes any function ``f(observed, simulated)`` to be minimized,
#   e.g. ``lambda obs, sim: -nse(obs, sim)``.
