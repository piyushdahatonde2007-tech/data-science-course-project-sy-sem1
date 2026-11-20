import io, json, platform
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import List, Optional

import joblib, numpy as np, pandas as pd, sklearn, xgboost
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config as C
from .datasets import Registry
from .features import CORE, FEATURE_COLS, clean_cell, extract_features
from . import nasa
from .alt_train import BUNDLE as ALT_BUNDLE, RES_JSON as ALT_RES, run_alt_training
from .nasa_train import BUNDLE as NASA_BUNDLE, FEAT_CSV, RES_JSON as NASA_RES, run_nasa_training
from .training import MODEL_NAMES, metrics, predict, run_training

S = SimpleNamespace(reg=None, res=None, bundle=None, pres=None, nres=None, nbundle=None, ntbl=None, nica=None, ares=None, abundle=None)


def versions():
    return dict(python=platform.python_version(), sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                numpy=np.__version__, pandas=pd.__version__)


def load_artifacts(reg):
    """Load trained models; retrain automatically if missing or built with different library versions."""
    mp, rp = C.ART_DIR / "models.joblib", C.ART_DIR / "results.json"
    try:
        bundle = joblib.load(mp)
        res = json.loads(rp.read_text())
        mm = lambda v: ".".join(str(v).split(".")[:2])   # compare major.minor only
        if any(mm(bundle["versions"].get(k)) != mm(versions()[k]) for k in ("sklearn", "xgboost", "numpy")):
            raise RuntimeError("library versions changed")
        return res, bundle
    except Exception as e:
        print(f"[startup] (re)training models: {e}. This takes ~2 minutes once.")
        return run_training(reg)


def load_nasa():
    """Load NASA-data models; retrain if missing or built with different library versions (features are cached)."""
    mm = lambda v: ".".join(str(v).split(".")[:2])
    try:
        b = joblib.load(C.ART_DIR / NASA_BUNDLE)
        r = json.loads((C.ART_DIR / NASA_RES).read_text())
        if any(mm(b["versions"].get(k)) != mm(versions()[k]) for k in ("sklearn", "xgboost", "numpy")):
            raise RuntimeError("library versions changed")
    except Exception as e:
        print(f"[startup] (re)training NASA models: {e}")
        r, b = run_nasa_training()
    tbl = pd.read_csv(C.ART_DIR / FEAT_CSV)
    return r, b, tbl


def load_alt():
    """Accelerated-life pack models; retrain from the cached cycle table if missing or built with different library versions."""
    mm = lambda v: ".".join(str(v).split(".")[:2])
    try:
        b = joblib.load(C.ART_DIR / ALT_BUNDLE)
        r = json.loads((C.ART_DIR / ALT_RES).read_text())
        if any(mm(b["versions"].get(k)) != mm(versions()[k]) for k in ("sklearn", "xgboost", "numpy")):
            raise RuntimeError("library versions changed")
    except Exception as e:
        print(f"[startup] (re)training accelerated-life models: {e}")
        r, b = run_alt_training()
    return r, b


@asynccontextmanager
async def lifespan(app):
    S.reg = Registry()
    S.res, S.bundle = load_artifacts(S.reg)
    S.nres, S.nbundle, S.ntbl = (None, None, None)
    if C.NASA_DIR.exists() or (C.ART_DIR / NASA_RES).exists():
        S.nres, S.nbundle, S.ntbl = load_nasa()
        S.nica = json.loads((C.ART_DIR / "nasa_ica_curves.json").read_text()) if (C.ART_DIR / "nasa_ica_curves.json").exists() else None
        S.ntbl = nasa.add_delta(S.ntbl)
    if (C.ART_DIR / "alt_cycles.csv").exists():
        S.ares, S.abundle = load_alt()
    S.pres = json.loads((C.ART_DIR / "paper_results.json").read_text()) if (C.ART_DIR / "paper_results.json").exists() else None
    yield


app = FastAPI(title="Battery RUL Console", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=C.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


# ---------- helpers ----------
def best():
    return S.res["meta"]["best_model"]


def score_table(tbl: pd.DataFrame, model: Optional[str] = None):
    """Predict cycle life + 80% interval for rows of a feature table using the final models."""
    model = model or best()
    if model not in S.bundle["models"]:
        raise HTTPException(400, f"unknown model '{model}'. Options: {MODEL_NAMES}")
    X = tbl[FEATURE_COLS].to_numpy(float)
    p = predict(S.bundle["models"][model], X)
    lo, hi = predict(S.bundle["lo"], X), predict(S.bundle["hi"], X)
    return p, np.minimum(lo, p), np.maximum(hi, p)


def record(bid, batch, actual, window, p, lo, hi, source):
    rul = p - window
    r = dict(battery_id=bid, batch=batch, predicted_cycle_life=float(p), lo=float(lo), hi=float(hi),
             predicted_rul=float(rul), tier=C.tier(rul), source=source)
    if actual is not None and not np.isnan(actual):
        r.update(actual_cycle_life=float(actual), actual_rul=float(actual - window), actual_tier=C.tier(actual - window))
    return r


def external_cells(did, model=None):
    t = S.reg.feat[did]
    p, lo, hi = score_table(t, model)
    return [record(t.battery_id[i], t.batch[i], t.cycle_life[i], t.max_cycle[i], p[i], lo[i], hi[i],
                   "final model (external)") for i in range(len(t))]


def dataset_cells(did):
    if S.reg.meta[did]["builtin"]:
        ids = set(S.reg.feat[did].battery_id)
        return [c for c in S.res["cells"] if c["battery_id"] in ids]
    return external_cells(did)


# ---------- routes ----------
@app.get("/api/health")
def health():
    return {"status": "ok", "best_model": best(), "datasets": len(S.reg.meta), "versions": versions()}


@app.get("/api/results")
def results():
    r = {k: v for k, v in S.res.items() if k not in ("cells", "oof")}
    r["models"] = MODEL_NAMES
    return r


@app.get("/api/oof")
def oof(model: Optional[str] = None):
    model = model or best()
    if model not in S.res["oof"]:
        raise HTTPException(400, f"unknown model; options: {MODEL_NAMES}")
    return {"model": model, **S.res["oof"][model]}


@app.get("/api/datasets")
def datasets():
    return list(S.reg.meta.values())


@app.get("/api/datasets/{did}")
def dataset(did: str):
    if did not in S.reg.meta:
        raise HTTPException(404, "dataset not found")
    t = S.reg.feat[did]
    cells = dataset_cells(did)
    out = dict(S.reg.meta[did])
    out["lives"] = [c["actual_cycle_life"] for c in cells if "actual_cycle_life" in c]
    out["tier_counts"] = pd.Series([c["tier"] for c in cells]).value_counts().to_dict()
    out["cv"] = S.res["cv"].get(did)
    if out["labeled"] and not out["builtin"]:
        a = np.array([c["actual_cycle_life"] for c in cells]); p = np.array([c["predicted_cycle_life"] for c in cells])
        out["external_validation"] = {"model": best(), **metrics(a, p), "n": len(a),
                                      "interval_coverage": float(np.mean([c["lo"] <= c["actual_cycle_life"] <= c["hi"] for c in cells])),
                                      "note": "Zero-shot: the final model was trained on MIT-Stanford data only."}
    return out


@app.get("/api/datasets/{did}/cells")
def cells(did: str, tier: Optional[str] = None, q: Optional[str] = None):
    if did not in S.reg.meta:
        raise HTTPException(404, "dataset not found")
    out = dataset_cells(did)
    if tier:
        out = [c for c in out if c["tier"] == tier]
    if q:
        out = [c for c in out if q.lower() in c["battery_id"].lower()]
    return out


@app.get("/api/datasets/{did}/cells/{bid}")
def cell(did: str, bid: str):
    if did not in S.reg.meta:
        raise HTTPException(404, "dataset not found")
    curves = S.reg.cell_curves(did, bid)
    if curves is None:
        raise HTTPException(404, "cell not found")
    t = S.reg.feat[did]
    row = t[t.battery_id == bid]
    base = next((c for c in dataset_cells(did) if c["battery_id"] == bid), None)
    live = {}
    if len(row):
        for m in MODEL_NAMES:
            p, lo, hi = score_table(row, m)
            live[m] = dict(predicted_cycle_life=float(p[0]), predicted_rul=float(p[0] - row.max_cycle.iloc[0]),
                           lo=float(lo[0]), hi=float(hi[0]))
    return {"battery_id": bid, "curves": curves, "record": base, "live": live,
            "features": {k: row.iloc[0][k] if pd.notna(row.iloc[0][k]) else None for k in FEATURE_COLS} if len(row) else {},
            "in_training": bool(S.reg.meta[did]["builtin"]),
            "note": "Final models were trained on this cell; use the out-of-fold record for an honest estimate."
            if S.reg.meta[did]["builtin"] else "External cell: never seen in training."}


class PredictIn(BaseModel):
    cycle: List[float]
    IR: List[float]; QC: List[float]; QD: List[float]
    Tavg: List[float]; Tmin: List[float]; Tmax: List[float]; chargetime: List[float]
    C1: Optional[float] = None; Q1: Optional[float] = None; C2: Optional[float] = None
    model: Optional[str] = None


@app.post("/api/predict")
def predict_live(b: PredictIn):
    d = b.model_dump()
    lens = {k: len(d[k]) for k in ["cycle"] + CORE}
    if len(set(lens.values())) != 1:
        raise HTTPException(422, f"all series must have equal length, got {lens}")
    df = pd.DataFrame({k: d[k] for k in ["cycle"] + CORE})
    s = clean_cell(df)
    if s is None:
        raise HTTPException(422, "need at least 10 valid cycles (finite values, QC>0, QD>0)")
    feats = extract_features(s, tuple(np.nan if d[k] is None else d[k] for k in ["C1", "Q1", "C2"]))
    tbl = pd.DataFrame([feats])
    p, lo, hi = score_table(tbl, b.model)
    mx = float(s["cycle"].max())
    warnings = []
    if not (45 <= mx <= 55):
        warnings.append(f"Models were trained on the first {C.EARLY_CYCLES} cycles; your data ends at cycle {mx:.0f}, so accuracy is unvalidated.")
    if d["C1"] is None:
        warnings.append("Charging-protocol fields (C1, Q1, C2) missing; median values were imputed.")
    return {"model": b.model or best(), "n_cycles_used": len(s["cycle"]),
            **record("live", "custom", None, mx, p[0], lo[0], hi[0], "live"), "warnings": warnings}


@app.post("/api/datasets/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > C.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file larger than {C.MAX_UPLOAD_MB} MB")
    try:
        df = pd.read_csv(io.BytesIO(raw))
        did = S.reg.add_upload(file.filename or "upload.csv", df)
    except Exception as e:
        raise HTTPException(422, f"could not read dataset: {e}")
    return dataset(did)


@app.delete("/api/datasets/{did}")
def delete_dataset(did: str):
    try:
        S.reg.delete_upload(did)
    except KeyError:
        raise HTTPException(404, "only uploaded datasets can be deleted")
    return {"deleted": did}


@app.get("/api/template.csv")
def template():
    p = C.ART_DIR / "template.csv"
    S.reg.raw[S.reg.raw.battery_id == S.reg.raw.battery_id.iloc[0]].to_csv(p, index=False)
    return FileResponse(p, filename="battery_dataset_template.csv", media_type="text/csv")


# ---------- NASA PCoE data ----------
def need_nasa():
    if S.nres is None:
        raise HTTPException(404, "NASA data is not installed (backend/data/nasa/cleaned_dataset missing)")


@app.get("/api/nasa/summary")
def nasa_summary():
    need_nasa()
    r = S.nres
    soh = {k: v for k, v in r["soh"].items() if k not in ("rows", "oof", "true")}
    rul = {k: v for k, v in r["rul"].items() if k != "rows"}
    return {"meta": r["meta"], "cells": r["cells"], "soh": soh, "rul": rul, "importance": r["importance"], "sp1": r["sp1"],
            "models": MODEL_NAMES}


@app.get("/api/nasa/oof")
def nasa_oof(model: Optional[str] = None, task: str = "soh"):
    need_nasa()
    model = model or S.nres["soh"]["best_model"]
    if model not in MODEL_NAMES:
        raise HTTPException(400, f"unknown model; options: {MODEL_NAMES}")
    rows = S.nres["soh"]["rows"]
    return {"model": model, "b": [x["b"] for x in rows], "k": [x["k"] for x in rows], "y": S.nres["soh"]["true"],
            "p": S.nres["soh"]["oof"][model]}


@app.get("/api/nasa/cells/{bid}")
def nasa_cell(bid: str):
    need_nasa()
    info = next((c for c in S.nres["cells"] if c["battery_id"] == bid), None)
    if info is None:
        raise HTTPException(404, "cell not found")
    t = S.ntbl[S.ntbl.battery_id == bid].sort_values("cycle_no")
    raw = [{"k": int(r.cycle_no), "cap": None if pd.isna(r.capacity) else round(float(r.capacity), 4), "ok": bool(r.cap_ok)} for r in t.itertuples()]
    soh = [x for x in S.nres["soh"]["rows"] if x["b"] == bid]
    rul = [x for x in S.nres["rul"]["rows"] if x["b"] == bid]
    return {"info": info, "raw_capacity": raw, "soh": soh, "rul": rul, "model": S.nres["soh"]["best_model"],
            "note": "Predictions are out-of-fold: the model that scored this cell never saw it in training." if soh else
            "This cell failed the quality gate, so it has no SOH label and no predictions."}


@app.get("/api/nasa/cells/{bid}/ica")
def nasa_ica(bid: str, n: int = Query(5, ge=1, le=8)):
    """Real dQ/dV curves of up to 8 cached charge cycles per cell (served from cache, so raw cycle files are not needed)."""
    need_nasa()
    if S.nica is None or bid not in S.nica["cells"]:
        raise HTTPException(404, "no cached ICA curves for this cell")
    cur = S.nica["cells"][bid]
    idx = sorted(set(np.linspace(0, len(cur) - 1, min(n, len(cur))).round().astype(int))) if cur else []
    return {"battery_id": bid, "window_v": S.nica["window_v"], "curves": [{"k": cur[i]["k"], "v": S.nica["v"], "ic": cur[i]["ic"]} for i in idx]}


class ChargeCycle(BaseModel):
    time: List[float]; voltage: List[float]; current: List[float]; temperature: List[float]


class NasaPredictIn(BaseModel):
    current_cycle: ChargeCycle
    reference_cycles: List[ChargeCycle]       # the cell's own first charge cycles (start-of-life reference), 1 to 10
    ambient_c: float
    model: Optional[str] = None


def _feats(c: ChargeCycle, amb):
    n = {len(c.time), len(c.voltage), len(c.current), len(c.temperature)}
    if len(n) != 1 or min(n) < 30:
        raise HTTPException(422, "each charge cycle needs equal-length time/voltage/current/temperature arrays with >= 30 samples")
    d = pd.DataFrame({"Time": c.time, "Voltage_measured": c.voltage, "Current_measured": c.current, "Temperature_measured": c.temperature})
    if not np.isfinite(d.to_numpy(float)).all():
        raise HTTPException(422, "charge cycle contains non-finite values")
    return nasa.features_from_df(d, amb)


@app.post("/api/nasa/predict-soh")
def nasa_predict_soh(b: NasaPredictIn):
    need_nasa()
    if not (1 <= len(b.reference_cycles) <= 10):
        raise HTTPException(422, "send 1 to 10 reference_cycles (the cell's first charge cycles)")
    model = b.model or S.nres["soh"]["best_model"]
    if model not in S.nbundle["soh"]:
        raise HTTPException(400, f"unknown model; options: {MODEL_NAMES}")
    cur = _feats(b.current_cycle, b.ambient_c)
    base = nasa.baseline_from_features([_feats(c, b.ambient_c) for c in b.reference_cycles])
    row = {**cur, **{k + "_d": cur[k] - base[k] for k in nasa.DELTA_BASE}}
    X = np.array([[row[c] for c in nasa.MODEL_COLS]], float)
    soh = float(np.clip(S.nbundle["soh"][model].predict(X)[0], 0, 120))
    warn = []
    if np.isnan(cur["ica_max"]):
        warn.append("Charge did not cover the 4.00-4.19 V ICA window; ICA features were imputed.")
    if len(b.reference_cycles) < nasa.REF_CYCLES:
        warn.append(f"Models used the median of {nasa.REF_CYCLES} start-of-life cycles; fewer gives a noisier reference.")
    return {"model": model, "soh_percent": soh, "warnings": warn,
            "note": "Cell-relative SOH: 100% = this cell's own early capacity. Typical leave-battery-out error is about 4 SOH points."}


# ---------- paper replication ----------
@app.get("/api/paper")
def paper():
    """Paper Table 2 vs. the paper's protocol re-run on our data, plus the paper's four models on real-ICA datasets."""
    if S.pres is None:
        raise HTTPException(404, "paper_results.json missing (run python -m app.paper_protocol)")
    four = ["XGBoost", "Random Forest", "Gradient Boosting", "SVR"]
    ica = {}
    if S.nres:
        ica["NASA PCoE: SOH estimation (21 cells)"] = {n: S.nres["soh"]["cv"][n] for n in four}
    if S.ares:
        ica["Accelerated-life packs: SOH estimation (25 packs)"] = {n: S.ares["soh"]["cv"][n] for n in four}
        ica["Accelerated-life packs: remaining life (16 packs)"] = {n: S.ares["rul"]["cv"][n] for n in four}
    return {**S.pres, "four_model_ica": ica, "models": four}


# ---------- NASA/UCF accelerated-life packs ----------
def need_alt():
    if S.ares is None:
        raise HTTPException(404, "accelerated-life data is not installed (backend/artifacts/alt_cycles.csv missing)")


@app.get("/api/alt/summary")
def alt_summary():
    need_alt()
    r = S.ares
    soh = {k: v for k, v in r["soh"].items() if k not in ("rows", "oof", "true")}
    rul = {k: v for k, v in r["rul"].items() if k != "rows"}
    return {"meta": r["meta"], "packs": r["packs"], "soh": soh, "rul": rul, "life_vs_load": r["life_vs_load"],
            "importance": r["importance"], "models": MODEL_NAMES}


@app.get("/api/alt/oof")
def alt_oof(task: str = "soh", model: Optional[str] = None):
    need_alt()
    if task == "soh":
        model = model or S.ares["soh"]["best_model"]
        if model not in MODEL_NAMES:
            raise HTTPException(400, f"unknown model; options: {MODEL_NAMES}")
        rows = S.ares["soh"]["rows"]
        return {"task": "soh", "model": model, "b": [x["b"] for x in rows], "k": [x["k"] for x in rows],
                "y": S.ares["soh"]["true"], "p": S.ares["soh"]["oof"][model]}
    if task == "rul":
        rows = S.ares["rul"]["rows"]
        return {"task": "rul", "model": S.ares["rul"]["best_model"], "b": [x["b"] for x in rows], "k": [x["k"] for x in rows],
                "y": [x["y"] for x in rows], "p": [x["p"] for x in rows]}
    raise HTTPException(400, "task must be soh or rul")


@app.get("/api/alt/packs/{pid}")
def alt_pack(pid: str, n: int = Query(5, ge=1, le=8)):
    need_alt()
    info = next((p for p in S.ares["packs"] if p["pack"] == pid), None)
    if info is None:
        raise HTTPException(404, "pack not found")
    cur = S.ares["ica"]["curves"].get(pid, [])
    idx = sorted(set(np.linspace(0, len(cur) - 1, n).round().astype(int))) if cur else []
    return {"info": info, "capacity": S.ares["capacity"][pid], "soh": [x for x in S.ares["soh"]["rows"] if x["b"] == pid],
            "rul": [x for x in S.ares["rul"]["rows"] if x["b"] == pid], "ica": {"v": S.ares["ica"]["v"], "curves": [cur[i] for i in idx]},
            "soh_model": S.ares["soh"]["best_model"], "rul_model": S.ares["rul"]["best_model"],
            "note": "Predictions are out-of-fold: the model that scored this pack never saw it in training."}


if C.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=C.FRONTEND_DIR, html=True), name="ui")
