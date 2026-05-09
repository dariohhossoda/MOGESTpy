"""
Hydrological network simulation using SMAP and Muskingum routing.

The module combines local rainfall-runoff estimates produced by ``SmapD`` with
upstream contributions routed by the linear Muskingum method. Each row in the
parameter table represents one subcatchment, and ``downstream_id`` defines the
directed network from upstream catchments to the outlet.

Expected input data
-------------------
``params`` must be a ``pandas.DataFrame`` with one row per subcatchment and the
following columns:

- ``id``: unique subcatchment identifier. Values must match the columns in the
  precipitation and evapotranspiration DataFrames.
- ``area``: drainage area in km2.
- ``str``, ``crec``, ``capc``, ``kkt``, ``k2t``, ``ai``, ``tuin`` and ``ebin``:
  SMAP parameters. ``tuin`` is expected as a percentage from 0 to 100 and is
  converted to the fraction expected by ``SmapD``.
- ``k`` and ``x``: Muskingum routing parameters. ``k`` must be non-negative,
  and it must be positive when the subcatchment receives upstream flow. ``x``
  must be between 0 and 0.5.
- ``downstream_id``: identifier of the next downstream subcatchment, or a null
  value for outlet subcatchments.

``precipitation`` and ``evapotranspiration`` must be DataFrames whose index
contains the simulation time steps and whose columns are the same
subcatchment identifiers found in ``params['id']``. Values are expected in
millimetres per time step. The returned DataFrame uses the common index shared
by precipitation and evapotranspiration and has one discharge column per
simulated subcatchment in m3/s.
"""

from __future__ import annotations

from collections import deque
from typing import Hashable

import numpy as np
import pandas as pd

from mogestpy.quantity.hydrological.smap import SmapD as Smap
from mogestpy.quantity.hydrological.muskingum import Muskingum

REQUIRED_PARAM_COLUMNS = (
    "id",
    "area",
    "str",
    "crec",
    "capc",
    "kkt",
    "k2t",
    "ai",
    "tuin",
    "ebin",
    "k",
    "x",
    "downstream_id",
)

NUMERIC_PARAM_COLUMNS = (
    "area",
    "str",
    "crec",
    "capc",
    "kkt",
    "k2t",
    "ai",
    "tuin",
    "ebin",
    "k",
    "x",
)

POSITIVE_PARAM_COLUMNS = ("area", "str", "kkt", "k2t")


def run_network(
    params: pd.DataFrame,
    precipitation: pd.DataFrame,
    evapotranspiration: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run SMAP with Muskingum routing over a hydrological network.

    Parameters
    ----------
    params : pd.DataFrame
        Model parameters per subcatchment. Required columns are defined by
        ``REQUIRED_PARAM_COLUMNS``. The ``id`` column must be unique and
        ``downstream_id`` must reference another ``id`` or a null outlet.
    precipitation : pd.DataFrame
        Precipitation time series in millimetres per time step. Columns must
        match all values in ``params['id']``.
    evapotranspiration : pd.DataFrame
        Potential evapotranspiration time series in millimetres per time step.
        Columns must match all values in ``params['id']``.
    verbose : bool, optional
        If True, prints progress messages. Default is False.

    Returns
    -------
    pd.DataFrame
        Simulated discharge in m3/s. The index is the common index shared by
        precipitation and evapotranspiration; columns are subcatchment IDs.

    Raises
    ------
    ValueError
        If the input DataFrames do not contain the required columns, contain
        invalid parameter values, have no common time steps or define an invalid
        network.
    """
    common_index = _validate_inputs(params, precipitation, evapotranspiration)

    models = _build_models(params)
    natural_flow = _compute_natural_flow(
        models,
        params,
        precipitation,
        evapotranspiration,
        common_index,
        verbose,
    )
    network_flow = _route_network(params, natural_flow, verbose)

    return pd.DataFrame(network_flow, index=common_index)


def _build_models(params: pd.DataFrame) -> dict[Hashable, Smap]:
    """Build one SMAP model for each subcatchment.

    Parameters
    ----------
    params : pd.DataFrame
        Validated parameter DataFrame with one row per subcatchment.

    Returns
    -------
    dict
        Mapping from subcatchment ID to initialized ``SmapD`` model.
    """
    models = {}

    for _, row in params.iterrows():
        sid = row["id"]

        models[sid] = Smap(
            Ad=float(row["area"]),
            Str=float(row["str"]),
            Crec=float(row["crec"]),
            Capc=float(row["capc"]),
            kkt=float(row["kkt"]),
            k2t=float(row["k2t"]),
            Ai=float(row["ai"]),
            Tuin=float(row["tuin"]) / 100,
            Ebin=float(row["ebin"]),
        )

    return models


def _compute_natural_flow(
    models: dict[Hashable, Smap],
    params: pd.DataFrame,
    precipitation: pd.DataFrame,
    evapotranspiration: pd.DataFrame,
    common_index: pd.Index,
    verbose: bool,
) -> dict[Hashable, np.ndarray]:
    """Compute local discharge for each subcatchment with SMAP.

    Parameters
    ----------
    models : dict
        Mapping from subcatchment ID to ``SmapD`` model.
    params : pd.DataFrame
        Validated parameter DataFrame.
    precipitation : pd.DataFrame
        Precipitation series with subcatchment IDs as columns.
    evapotranspiration : pd.DataFrame
        Evapotranspiration series with subcatchment IDs as columns.
    common_index : pd.Index
        Time steps shared by precipitation and evapotranspiration.
    verbose : bool
        If True, prints progress messages.

    Returns
    -------
    dict
        Mapping from subcatchment ID to local discharge arrays in m3/s.
    """
    natural_flow = {}

    for sid in params["id"]:
        rain = pd.to_numeric(precipitation.loc[common_index, sid])
        etp = pd.to_numeric(evapotranspiration.loc[common_index, sid])

        try:
            natural_flow[sid] = np.asarray(
                models[sid].run_to_list(rain, etp), dtype=float
            )
        except Exception as e:
            raise RuntimeError(f"Error computing natural flow for basin {sid}") from e

        if not np.isfinite(natural_flow[sid]).all():
            raise ValueError(f"Natural flow for basin {sid} contains invalid values.")

    return natural_flow


def _route_network(
    params: pd.DataFrame,
    natural_flow: dict[Hashable, np.ndarray],
    verbose: bool,
) -> dict[Hashable, np.ndarray]:
    """Route local and upstream flows according to the basin network.

    Parameters
    ----------
    params : pd.DataFrame
        Validated parameter DataFrame.
    natural_flow : dict
        Mapping from subcatchment ID to local discharge arrays.
    verbose : bool
        If True, prints progress messages.

    Returns
    -------
    dict
        Mapping from subcatchment ID to routed discharge arrays in m3/s.
    """
    order = _topological_order(params)
    params_by_id = params.set_index("id", drop=False)
    routed_flow = {}

    for sid in order:
        if verbose:
            print(f"[INFO] Processing basin {sid}")

        q_local = natural_flow[sid]
        n = len(q_local)

        row = params_by_id.loc[sid]
        k = float(row["k"])
        x = float(row["x"])

        upstream_ids = _get_upstream_ids(params, sid)
        q_upstream = np.zeros(n)

        for uid in upstream_ids:
            q_upstream += routed_flow[uid]

        if upstream_ids and k <= 0:
            raise ValueError(
                "params['k'] must be positive for basins that receive "
                f"upstream flow. Invalid basin: {sid}."
            )

        routed_flow[sid] = _compute_total_flow(q_local, q_upstream, k, x)

    return routed_flow


def _validate_inputs(
    params: pd.DataFrame,
    precipitation: pd.DataFrame,
    evapotranspiration: pd.DataFrame,
) -> pd.Index:
    """Validate the parameter table and forcing time series.

    Parameters
    ----------
    params : pd.DataFrame
        Parameter table for the hydrological network.
    precipitation : pd.DataFrame
        Precipitation forcing time series.
    evapotranspiration : pd.DataFrame
        Evapotranspiration forcing time series.

    Returns
    -------
    pd.Index
        Time index shared by precipitation and evapotranspiration.
    """
    _validate_params(params)
    _validate_timeseries_columns(params, precipitation, "precipitation")
    _validate_timeseries_columns(
        params,
        evapotranspiration,
        "evapotranspiration",
    )

    common_index = precipitation.index.intersection(evapotranspiration.index)
    if common_index.empty:
        raise ValueError(
            "Precipitation and evapotranspiration DataFrames must share at "
            "least one time step."
        )

    ids = params["id"].tolist()
    for name, data in (
        ("precipitation", precipitation),
        ("evapotranspiration", evapotranspiration),
    ):
        values = data.loc[common_index, ids]
        if values.isna().any().any():
            invalid_columns = values.columns[values.isna().any()].tolist()
            raise ValueError(
                f"{name} contains missing values for columns: {invalid_columns}."
            )

        numeric_values = values.apply(pd.to_numeric, errors="coerce")
        if numeric_values.isna().any().any():
            invalid_columns = numeric_values.columns[
                numeric_values.isna().any()
            ].tolist()
            raise ValueError(
                f"{name} contains non-numeric values for columns: {invalid_columns}."
            )

    return common_index


def _validate_params(params: pd.DataFrame) -> None:
    """Validate the parameter DataFrame structure and values.

    Parameters
    ----------
    params : pd.DataFrame
        Parameter table to validate.

    Raises
    ------
    ValueError
        If required columns, parameter values or network references are invalid.
    """
    if params.empty:
        raise ValueError("params must contain at least one subcatchment.")

    missing_columns = [
        column for column in REQUIRED_PARAM_COLUMNS if column not in params.columns
    ]
    if missing_columns:
        raise ValueError(f"params is missing required columns: {missing_columns}.")

    if params["id"].isna().any():
        raise ValueError("params['id'] must not contain null values.")

    duplicated_ids = params.loc[params["id"].duplicated(), "id"].tolist()
    if duplicated_ids:
        raise ValueError(f"params['id'] contains duplicated values: {duplicated_ids}.")

    for column in NUMERIC_PARAM_COLUMNS:
        values = pd.to_numeric(params[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"params['{column}'] must contain numeric values.")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"params['{column}'] must contain finite values.")

    for column in POSITIVE_PARAM_COLUMNS:
        values = pd.to_numeric(params[column], errors="coerce")
        if (values <= 0).any():
            raise ValueError(f"params['{column}'] must contain positive values.")

    non_negative_columns = ("crec", "capc", "ai", "tuin", "ebin", "k")
    for column in non_negative_columns:
        values = pd.to_numeric(params[column], errors="coerce")
        if (values < 0).any():
            raise ValueError(f"params['{column}'] must not contain negative values.")

    tuin = pd.to_numeric(params["tuin"], errors="coerce")
    if (tuin > 100).any():
        raise ValueError("params['tuin'] must be a percentage between 0 and 100.")

    x = pd.to_numeric(params["x"], errors="coerce")
    if ((x < 0) | (x > 0.5)).any():
        raise ValueError("params['x'] must be between 0 and 0.5.")

    ids = set(params["id"])
    unknown_downstream = [
        downstream_id
        for downstream_id in params["downstream_id"].dropna()
        if downstream_id not in ids
    ]
    if unknown_downstream:
        raise ValueError(
            "params['downstream_id'] contains IDs not present in params['id']: "
            f"{unknown_downstream}."
        )

    self_loops = params.loc[params["id"] == params["downstream_id"], "id"].tolist()
    if self_loops:
        raise ValueError(f"Network contains self-loops for IDs: {self_loops}.")

    _topological_order(params)


def _validate_timeseries_columns(
    params: pd.DataFrame,
    data: pd.DataFrame,
    name: str,
) -> None:
    """Validate time series columns against subcatchment IDs.

    Parameters
    ----------
    params : pd.DataFrame
        Validated parameter DataFrame.
    data : pd.DataFrame
        Time series DataFrame to validate.
    name : str
        Human-readable DataFrame name for error messages.
    """
    if data.empty:
        raise ValueError(f"{name} must contain at least one time step.")

    duplicated_columns = data.columns[data.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(f"{name} contains duplicated columns: {duplicated_columns}.")

    missing_columns = [sid for sid in params["id"] if sid not in data.columns]
    if missing_columns:
        raise ValueError(
            f"{name} is missing time series columns for IDs: {missing_columns}."
        )


def _topological_order(params: pd.DataFrame) -> list[Hashable]:
    """Return subcatchment IDs from headwaters to outlets.

    Parameters
    ----------
    params : pd.DataFrame
        Parameter DataFrame with ``id`` and ``downstream_id`` columns.

    Returns
    -------
    list
        Subcatchment IDs ordered so upstream basins are processed before their
        downstream receivers.
    """
    ids = params["id"].tolist()
    canonical_id = {sid: sid for sid in ids}
    downstream_by_id = {sid: [] for sid in ids}
    indegree = {sid: 0 for sid in ids}

    for _, row in params.iterrows():
        sid = row["id"]
        downstream_id = row["downstream_id"]
        if pd.isna(downstream_id):
            continue

        downstream_id = canonical_id[downstream_id]
        downstream_by_id[sid].append(downstream_id)
        indegree[downstream_id] += 1

    queue = deque(sid for sid in ids if indegree[sid] == 0)
    order = []

    while queue:
        sid = queue.popleft()
        order.append(sid)

        for downstream_id in downstream_by_id[sid]:
            indegree[downstream_id] -= 1
            if indegree[downstream_id] == 0:
                queue.append(downstream_id)

    if len(order) != len(ids):
        cyclic_ids = [sid for sid in ids if indegree[sid] > 0]
        raise ValueError(f"Network contains at least one cycle: {cyclic_ids}.")

    return order


def _compute_total_flow(
    q_local: np.ndarray,
    q_upstream: np.ndarray,
    k: float,
    x: float,
) -> np.ndarray:
    """Combine local and upstream flow using Muskingum routing.

    Parameters
    ----------
    q_local : np.ndarray
        Local discharge produced by SMAP in m3/s.
    q_upstream : np.ndarray
        Sum of routed upstream discharges in m3/s.
    k : float
        Muskingum storage coefficient in time-step units.
    x : float
        Muskingum weighting factor, expected between 0 and 0.5.

    Returns
    -------
    np.ndarray
        Total discharge in m3/s for the subcatchment outlet.
    """

    if q_local.shape != q_upstream.shape:
        raise ValueError("Input series must have the same length")

    if q_local.size == 0:
        return np.array([])

    if not np.isfinite(q_local).all() or not np.isfinite(q_upstream).all():
        raise ValueError("Input series must contain only finite values.")

    if not np.any(q_upstream):
        return q_local.copy()

    routed = np.asarray(
        Muskingum.downstream_routing(q_upstream, k, x, dt=1),
        dtype=float,
    )

    return q_local + routed


def _get_upstream_ids(params: pd.DataFrame, basin_id: Hashable) -> list[Hashable]:
    """Return the direct upstream subcatchments for a basin.

    Parameters
    ----------
    params : pd.DataFrame
        Parameter DataFrame with ``id`` and ``downstream_id`` columns.
    basin_id : hashable
        Target subcatchment ID.

    Returns
    -------
    list
        IDs whose ``downstream_id`` equals ``basin_id``.
    """
    return params.loc[params["downstream_id"] == basin_id, "id"].tolist()
