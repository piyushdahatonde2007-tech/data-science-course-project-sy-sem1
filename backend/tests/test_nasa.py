import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app import config as C, nasa
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_capacity_cleaning_removes_spike_and_zero():
    cap = np.array([1.8, 1.8, 1.79, 2.9, 1.78, 0.0, 1.77, 1.77, 1.76, 1.76])
    keep, _ = nasa.clean_capacity(cap)
    assert not keep[3] and not keep[5] and keep.sum() == 8


def test_quality_gate_rejects_ramp_and_accepts_stable():
    stable = 1.8 - 0.004 * np.arange(60)
    ramp = np.linspace(0.9, 1.6, 40)   # capacity drifting upward: not a valid fade trajectory
    assert nasa.cell_quality(stable)["ok"] and not nasa.cell_quality(ramp)["ok"]


def test_nasa_summary_and_cells(client):
    s = client.get("/api/nasa/summary").json()
    assert s["meta"]["cells_total"] == 34 and s["meta"]["cells_ok"] == 21 and s["meta"]["data_files"] == 7565
    c = client.get("/api/nasa/cells/B0005").json()
    assert len(c["soh"]) > 100 and c["info"]["ok"]
    assert client.get("/api/nasa/cells/B0041").json()["soh"] == []
    assert client.get("/api/nasa/cells/NOPE").status_code == 404
    o = client.get("/api/nasa/oof").json()
    assert len(o["y"]) == len(o["p"]) == len(o["b"])


def test_ica_curve_endpoint(client):
    r = client.get("/api/nasa/cells/B0005/ica?n=3").json()
    assert 1 <= len(r["curves"]) <= 3 and len(r["curves"][0]["v"]) == len(r["curves"][0]["ic"])


def _cycle(fn):
    d = pd.read_csv(C.NASA_DIR / "data" / fn)
    return {"time": d.Time.tolist(), "voltage": d.Voltage_measured.tolist(), "current": d.Current_measured.tolist(),
            "temperature": d.Temperature_measured.tolist()}


@pytest.mark.skipif(not (C.NASA_DIR / "data").exists(), reason="raw NASA cycle files not installed (lean zip)")
def test_predict_soh_roundtrip_and_validation(client):
    t = pd.read_csv(C.ART_DIR / "nasa_features.csv")
    b = t[t.battery_id == "B0005"].sort_values("cycle_no")
    body = {"current_cycle": _cycle(b.charge_file.iloc[-1]), "reference_cycles": [_cycle(f) for f in b.charge_file.iloc[:5]], "ambient_c": 24}
    r = client.post("/api/nasa/predict-soh", json=body)
    assert r.status_code == 200 and 40 < r.json()["soh_percent"] < 100   # late-life cell: clearly degraded
    body["reference_cycles"] = []
    assert client.post("/api/nasa/predict-soh", json=body).status_code == 422
