"""
Tests for the warnings of the SIHQUAL module.
"""

import os
import tempfile

import pandas as pd
import pytest

from mogestpy.quantity.hydrodynamic import sihqual
from mogestpy.quantity.hydrodynamic.sihqual import SIHQUAL


def test_save_results_warns_and_returns_false_on_error(capsys):
    model = SIHQUAL.create_simple_channel(length=1000, simulation_time=3600)
    results = pd.DataFrame({"Q": [1.0, 2.0]})

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "missing", "results.xlsx")
        with pytest.warns(UserWarning, match="Error saving results"):
            assert model.save_results(results, filename) is False

    assert capsys.readouterr().out == ""


def test_run_simulation_is_deprecated(monkeypatch, capsys):
    class FakeModel:
        def run(self):
            return "results"

    monkeypatch.setattr(
        SIHQUAL, "create_simple_channel", staticmethod(lambda **kwargs: FakeModel())
    )

    with pytest.warns(DeprecationWarning, match="run_simulation is deprecated"):
        assert sihqual.run_simulation() == "results"

    assert capsys.readouterr().out == ""
