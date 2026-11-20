"""Pipeline plumbing test with a tiny SYNTHETIC fixture (not real battery data)."""
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import convert_severson as cs
from app.datasets import REQUIRED
from app.features import feature_table


def fake_batch(prefix, n_cells, rng):
    out = {}
    for i in range(n_cells):
        n = 60
        out[f"{prefix}c{i}"] = {
            "cycle_life": np.array([[600.0 + 50 * i]]), "charge_policy": f"{3 + i}C({20 + 10 * i}%)-4C",
            "summary": {"cycle": np.arange(1, n + 1), "IR": 0.016 + 1e-5 * np.arange(n), "QC": 1.07 - 1e-4 * np.arange(n),
                        "QD": 1.07 - 1e-4 * np.arange(n) + rng.normal(0, 1e-4, n), "Tavg": 31 + rng.normal(0, .1, n),
                        "Tmin": 30 + 0 * np.arange(n), "Tmax": 33 + 0 * np.arange(n), "chargetime": 12 + 0.01 * np.arange(n)}}
    out[f"{prefix}bad"] = {"cycle_life": np.array([[np.nan]]), "summary": {}}
    return out


def test_convert_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for k, pre in enumerate(["b1", "b2"]):
        p = tmp_path / f"{pre}.pkl"
        p.write_bytes(pickle.dumps(fake_batch(pre, 3, rng)))
        paths.append(str(p))
    out = tmp_path / "m.csv"
    cs.main([*paths, "--out", str(out), "--cycles", "50", "--exclude", "b2c2"])
    df = pd.read_csv(out)
    assert set(REQUIRED) <= set(df.columns) and {"C1", "Q1", "C2", "cycle_life"} <= set(df.columns)
    assert df.battery_id.nunique() == 5 and df.cycle.max() == 50
    assert df[df.battery_id == "b1c1"].C1.iloc[0] == 4.0
    t = feature_table(df)
    assert len(t) == 5 and t.batch.tolist().count("1") == 3
