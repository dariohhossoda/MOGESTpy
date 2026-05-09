import numpy as np
import pandas as pd
import pytest

from mogestpy.quantity.hydrological.routing import run_network
from mogestpy.quantity.hydrological.smap import SmapD as Smap


def test_routing_module_is_available_as_optional_hydrological_module():
    from mogestpy.quantity.hydrological import routing

    assert routing.run_network is run_network


def _params() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["headwater", "outlet"],
            "area": [513.37, 400.0],
            "str": [1559.91, 1200.0],
            "crec": [0.25, 0.2],
            "capc": [40.0, 40.0],
            "kkt": [83.63, 70.0],
            "k2t": [6.49, 5.5],
            "ai": [2.5, 2.5],
            "tuin": [50.0, 45.0],
            "ebin": [11.05, 8.0],
            "k": [1.0, 1.0],
            "x": [0.2, 0.2],
            "downstream_id": ["outlet", np.nan],
        }
    )


def _forcing(ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    precipitation = pd.DataFrame(
        {
            sid: [0.0, 5.0, 0.0, 12.0]
            for sid in ids
        },
        index=index,
    )
    evapotranspiration = pd.DataFrame(
        {
            sid: [4.5, 4.5, 4.5, 4.5]
            for sid in ids
        },
        index=index,
    )
    return precipitation, evapotranspiration


def test_run_network_keeps_outlet_without_downstream_edge():
    params = _params().iloc[[1]].copy()
    precipitation, evapotranspiration = _forcing(["outlet"])

    result = run_network(params, precipitation, evapotranspiration)

    smap = Smap(
        Ad=400.0,
        Str=1200.0,
        Crec=0.2,
        Capc=40.0,
        kkt=70.0,
        k2t=5.5,
        Ai=2.5,
        Tuin=0.45,
        Ebin=8.0,
    )
    expected = smap.run_to_list(
        precipitation["outlet"],
        evapotranspiration["outlet"],
    )

    assert result.index.equals(precipitation.index)
    assert result.columns.tolist() == ["outlet"]
    assert result["outlet"].tolist() == pytest.approx(expected)


def test_run_network_routes_upstream_flow_to_downstream_basin():
    params = _params()
    precipitation, evapotranspiration = _forcing(["headwater", "outlet"])

    result = run_network(params, precipitation, evapotranspiration)

    assert result.columns.tolist() == ["headwater", "outlet"]
    assert len(result) == len(precipitation)
    assert (result["outlet"] >= result["headwater"]).any()


def test_run_network_rejects_missing_forcing_column():
    params = _params()
    precipitation, evapotranspiration = _forcing(["headwater", "outlet"])
    precipitation = precipitation.drop(columns=["headwater"])

    with pytest.raises(ValueError, match="precipitation is missing"):
        run_network(params, precipitation, evapotranspiration)


def test_run_network_rejects_network_cycles():
    params = _params()
    params.loc[params["id"] == "outlet", "downstream_id"] = "headwater"
    precipitation, evapotranspiration = _forcing(["headwater", "outlet"])

    with pytest.raises(ValueError, match="cycle"):
        run_network(params, precipitation, evapotranspiration)
