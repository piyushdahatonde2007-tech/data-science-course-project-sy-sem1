import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import config as C


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_results(client):
    assert client.get("/api/health").json()["status"] == "ok"
    r = client.get("/api/results").json()
    assert r["meta"]["best_model"] in r["models"] and "cross_batch" in r


def test_datasets(client):
    ids = {d["id"] for d in client.get("/api/datasets").json()}
    assert {"mit_all", "mit_b1", "mit_b2", "mit_b3"} <= ids
    cells = client.get("/api/datasets/mit_b2/cells").json()
    assert len(cells) == 48 and all(c["source"] == "out-of-fold" for c in cells)


def test_cell_and_predict(client):
    c = client.get("/api/datasets/mit_all/cells/b1c0").json()
    assert set(c["live"]) >= {"XGBoost", "SVR"} and len(c["curves"]["cycle"]) >= 49
    body = {k: c["curves"][k.lower()] for k in ["cycle", "IR", "QC", "QD", "Tavg", "chargetime"]}
    body.update(Tmin=c["curves"]["tavg"], Tmax=c["curves"]["tavg"])
    r = client.post("/api/predict", json=body)
    assert r.status_code == 200 and r.json()["lo"] <= r.json()["predicted_cycle_life"] <= r.json()["hi"]


def test_predict_validation(client):
    bad = {k: [1.0] * 12 for k in ["cycle", "IR", "QC", "QD", "Tavg", "Tmin", "Tmax", "chargetime"]}
    bad["IR"] = [1.0] * 5
    assert client.post("/api/predict", json=bad).status_code == 422
    short = {k: [1.0] * 4 for k in bad}
    assert client.post("/api/predict", json=short).status_code == 422


def test_upload_roundtrip(client):
    df = pd.read_csv(C.RAW_CSV)
    sub = df[df.battery_id.isin(df.battery_id.unique()[:12])]
    r = client.post("/api/datasets/upload", files={"file": ("my set.csv", sub.to_csv(index=False), "text/csv")})
    assert r.status_code == 200 and r.json()["n_cells"] == 12 and "external_validation" in r.json()
    assert client.delete(f"/api/datasets/{r.json()['id']}").status_code == 200
    assert client.post("/api/datasets/upload", files={"file": ("x.csv", "a,b\n1,2", "text/csv")}).status_code == 422


def test_protocol_results(client):
    r = client.get("/api/results").json()
    assert len(r["protocols"]) >= 10 and set(r["protocol_cv"]) == set(r["models"])
    c = client.get("/api/datasets/mit_all/cells").json()[0]
    assert "protocol" in c and "c1" in c


def test_paper_replication(client):
    p = client.get("/api/paper").json()
    assert set(p["paper"]) == set(p["models"]) and len(p["seeds"]) >= 5
    assert p["results"]["XGBoost"]["mean"]["R2"] > 0.3 and p["n_test"] == 58
    assert "NASA PCoE: SOH estimation (21 cells)" in p["four_model_ica"]
