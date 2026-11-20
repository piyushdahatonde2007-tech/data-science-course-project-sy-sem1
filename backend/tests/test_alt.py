import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app import alt
from app.main import app


def synthetic_pack(tmp_path):
    """Tiny SYNTHETIC pack log (not real data): rest, charge, reference discharge, charge, regular discharge."""
    rows, t = [], 0.0
    def seg(mode, n, vch=0.0, vload=np.nan, i=np.nan, mt=np.nan, dt=1.0):
        nonlocal t
        for j in range(n):
            rows.append(dict(start_time="2022-07-19 15:03:00", time=t, mode=mode, voltage_charger=vch if not callable(vch) else vch(j, n),
                             temperature_battery=25.0, voltage_load=vload(j, n) if callable(vload) else vload, current_load=i,
                             temperature_mosfet=np.nan, temperature_resistor=np.nan, mission_type=mt))
            t += dt
    seg(0, 30, vch=8.2)
    seg(1, 400, vch=lambda j, n: 6.5 + 2.1 * j / n)
    seg(0, 30, vch=8.6)
    seg(-1, 1500, vload=lambda j, n: 8.3 - 3.4 * j / n, i=2.5, mt=0, dt=2.0)      # reference discharge: 2.5 A, V 8.3 -> 4.9
    seg(1, 400, vch=lambda j, n: 6.5 + 2.1 * j / n)
    seg(0, 30, vch=8.6)
    seg(-1, 300, vload=lambda j, n: 8.6 - 3.0 * j / n, i=19.0, mt=1)
    p = tmp_path / "battery99.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_extractor_on_synthetic_pack(tmp_path):
    df, curves = alt.extract_pack(synthetic_pack(tmp_path))
    assert len(df) == 2 and df.mission.tolist() == [0, 1]
    assert abs(df.cap_ah.iloc[0] - 2.5 * 3000 / 3600) < 0.01              # integral of 2.5 A over 3000 s
    assert len(curves) == 1 and len(curves[0]["ic"]) == len(alt.VPLOT)
    assert df.ch_dur_s.iloc[1] > 300 and df.rest_v.iloc[1] > 8


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_alt_api(client):
    s = client.get("/api/alt/summary").json()
    assert s["meta"]["packs"] == 26 and s["meta"]["cycles"] == 9614 and len(s["packs"]) == 26
    assert s["life_vs_load"]["slope"] < 0                                  # higher load, shorter life
    p = client.get("/api/alt/packs/01").json()
    assert len(p["capacity"]) == 862 and p["soh"] and 1 <= len(p["ica"]["curves"]) <= 5
    assert client.get("/api/alt/packs/zz").status_code == 404
    o = client.get("/api/alt/oof?task=rul").json()
    assert len(o["y"]) == len(o["p"]) == len(o["b"]) > 1000
    assert client.get("/api/alt/oof?task=nope").status_code == 400
