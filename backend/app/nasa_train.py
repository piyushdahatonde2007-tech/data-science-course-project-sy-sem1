"""Train and evaluate on the NASA PCoE data: SOH estimation from charge-curve ICA features (21 cells passing the quality
gate) and remaining-useful-life to 80% SOH (cells that reach it). All validation is leave-battery-out."""
import json, platform, time
import numpy as np
import pandas as pd
import joblib, sklearn, xgboost
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from . import config as C
from . import nasa
from .nasa import CHARGE_COLS, DELTA_COLS, FEATURE_COLS, ICA_COLS, MODEL_COLS, TEMP_COLS
from .training import MODEL_NAMES, make_model

FEAT_CSV, RES_JSON, BUNDLE = "nasa_features.csv", "nasa_results.json", "nasa_models.joblib"


def mk(name):
    """Same model family as the MIT pipeline, with the target standardised (SOH % / cycles are not log-scaled)."""
    return TransformedTargetRegressor(regressor=make_model(name), transformer=StandardScaler())


def met(y, p):
    e = y - p
    return dict(R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), RMSE=float(np.sqrt(np.mean(e ** 2))),
                MAE=float(np.mean(np.abs(e))))


def agg(runs):
    out = {}
    for k in runs[0]:
        v = [r[k] for r in runs]
        out[k], out[k + "_std"] = float(np.mean(v)), float(np.std(v))
    return out


def group_cv(name, X, y, groups, repeats=C.CV_REPEATS, folds=C.CV_FOLDS):
    """Leave-battery-out CV: whole batteries are held out together. Returns metrics (mean/std) and repeat-0 OOF preds."""
    ug, runs, first = np.unique(groups), [], None
    for r in range(repeats):
        parts = np.array_split(np.random.default_rng(C.SEED + r).permutation(ug), min(folds, len(ug)))
        oof = np.zeros(len(y))
        for part in parts:
            te = np.isin(groups, part)
            oof[te] = mk(name).fit(X[~te], y[~te]).predict(X[te])
        runs.append(met(y, oof))
        if r == 0:
            first = oof
    return agg(runs), first


def build_table(out_dir, log=print):
    """Paired charge/discharge cycles + charge-curve features for every cycle (cached)."""
    cache = out_dir / FEAT_CSV
    if cache.exists():
        return pd.read_csv(cache)
    t0 = time.time()
    m = nasa.load_metadata()
    pairs = nasa.pair_cycles(m)
    log(f"[nasa] extracting features from {len(pairs)} charge files…")
    feats = Parallel(n_jobs=-1, batch_size=16)(
        delayed(nasa.charge_features)(C.NASA_DIR / "data" / r.charge_file, r.ambient_c) for r in pairs.itertuples())
    tbl = pd.concat([pairs.reset_index(drop=True), pd.DataFrame(feats).drop(columns=["ambient_c"])], axis=1)
    tbl.to_csv(cache, index=False)
    log(f"[nasa] features done in {time.time() - t0:.0f}s")
    return tbl


def sp1_summary():
    """Single reference test (Arbin export): step table, voltage/current profile, initial capacity and a real dQ/dV curve."""
    if C.SP1_XLSX is None or not C.SP1_XLSX.exists():
        return None
    from scipy.signal import savgol_filter
    d = pd.read_excel(C.SP1_XLSX, sheet_name=0)
    steps = []
    for si, g in d.groupby("Step_Index"):
        i = float(g["Current(A)"].median())
        mode = "CC charge" if i > 0.5 else "CV charge" if i > 0.02 else "CC discharge" if i < -0.5 else "rest"
        steps.append(dict(step=int(si), mode=mode, rows=int(len(g)), hours=round(float((g["Test_Time(s)"].max() - g["Test_Time(s)"].min()) / 3600), 3),
                          v_start=round(float(g["Voltage(V)"].iloc[0]), 3), v_end=round(float(g["Voltage(V)"].iloc[-1]), 3), current_a=round(i, 3)))
    ds = d.iloc[::25]
    cc = d[(d["Current(A)"] > 0.5) & (d["Step_Index"] == d[d["Current(A)"] > 0.5]["Step_Index"].iloc[0])]
    V = np.maximum.accumulate(cc["Voltage(V)"].to_numpy(float))
    Vu, ix = np.unique(V, return_index=True)
    grid = np.round(np.arange(3.60, 4.19, 0.002), 4)
    q = np.interp(grid, Vu, cc["Charge_Capacity(Ah)"].to_numpy(float)[ix])
    ic = savgol_filter(np.gradient(q, grid), 21, 3)
    return dict(file=C.SP1_XLSX.name, rows=int(len(d)), steps=steps, initial_capacity_ah=round(float(d["Discharge_Capacity(Ah)"].max()), 3),
                profile=dict(hours=[round(float(x) / 3600, 3) for x in ds["Test_Time(s)"]], v=[round(float(x), 4) for x in ds["Voltage(V)"]],
                             i=[round(float(x), 3) for x in ds["Current(A)"]]),
                ica=dict(v=grid.tolist(), ic=[round(float(x), 4) for x in ic]),
                note="One reference test on a single cell (1 A charge to 4.2 V, CV hold, rest, 1 A discharge to 2.5 V). Too short to train on; shown as reference data.")


def run_nasa_training(out_dir=C.ART_DIR, log=print):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl = nasa.add_delta(build_table(out_dir, log))
    m = nasa.load_metadata()
    # ---- per-cell quality table
    cells = []
    for b, g in m.groupby("battery_id", sort=True):
        q = nasa.cell_quality(g[g.type == "discharge"].cap.to_numpy(float))
        sub = tbl[tbl.battery_id == b]
        cells.append(dict(battery_id=b, ambient_c=sorted(set(float(x) for x in g.ambient_temperature)), n_cycles=q["n_discharges"],
                          n_pairs=int(len(sub)), ref_capacity=q["ref_capacity"], ok=q["ok"], reason=q["reason"],
                          eol80_cycle=q["eol80_cycle"], eol70_cycle=q["eol_cycle"]))
    cdf = pd.DataFrame(cells).set_index("battery_id")
    # ---- SOH task
    d = tbl[tbl.cell_ok & tbl.cap_ok & tbl.soh.notna()].reset_index(drop=True)
    X, y, grp = d[MODEL_COLS].to_numpy(float), d.soh.to_numpy(float), d.battery_id.to_numpy()
    cov = {"ica_window_coverage": float(d[ICA_COLS].notna().all(axis=1).mean()), "rows": int(len(d)), "cells": int(d.battery_id.nunique())}
    res = {"meta": {"cv_folds": C.CV_FOLDS, "cv_repeats": C.CV_REPEATS, "target_soh": "capacity / reference capacity x 100 (cell-relative)",
                    "ica_window_v": [nasa.V_LO, nasa.V_HI], "features": len(MODEL_COLS), "ref_cycles": nasa.REF_CYCLES, **cov,
                    "data_files": int(len(m)), "cells_total": int(m.battery_id.nunique()), "cells_ok": int(cdf.ok.sum())},
           "cells": json.loads(cdf.reset_index().to_json(orient="records")), "soh": {"cv": {}}, "rul": {"cv": {}}}
    oof = {}
    for n in MODEL_NAMES:
        res["soh"]["cv"][n], oof[n] = group_cv(n, X, y, grp)
        log(f"[soh] {n:18s} R2={res['soh']['cv'][n]['R2']:.3f} RMSE={res['soh']['cv'][n]['RMSE']:.2f}")
    best = min(MODEL_NAMES, key=lambda n: res["soh"]["cv"][n]["RMSE"])
    res["soh"]["best_model"] = best
    res["soh"]["rows"] = [dict(b=r.battery_id, k=int(r.cycle_no), y=round(float(r.soh), 2), p=round(float(oof[best][i]), 2))
                          for i, r in enumerate(d.itertuples())]
    res["soh"]["oof"] = {n: [round(float(v), 2) for v in oof[n]] for n in MODEL_NAMES}
    res["soh"]["true"] = [round(float(v), 2) for v in y]
    res["soh"]["per_cell"] = [dict(battery_id=b, n=int(len(s)), **met(s.y.to_numpy(), s.p.to_numpy())) for b, s in
                              pd.DataFrame(res["soh"]["rows"]).groupby("b") for s in [s]]
    # ---- ablation: do ICA features add information beyond cycle count / charge statistics?
    d["cycle_feat"] = d.cycle_no.astype(float)
    dl = lambda cols: [c + "_d" for c in cols]
    other = CHARGE_COLS + ["t_mean", "t_max", "t_rise"]
    sets = {"absolute features only": FEATURE_COLS, "absolute ICA only (+ambient)": ICA_COLS + ["ambient_c"],
            "absolute charge/CV stats only": CHARGE_COLS + TEMP_COLS, "cycle number only (+ambient)": ["cycle_feat", "ambient_c"],
            "Δ from own start, all (+ambient)": DELTA_COLS + ["ambient_c"], "Δ ICA only (+ambient)": dl(ICA_COLS) + ["ambient_c"],
            "Δ charge/CV only (+ambient)": dl(other) + ["ambient_c"], "absolute + Δ (headline)": MODEL_COLS}
    res["soh"]["ablation"] = {best: {k: (res["soh"]["cv"][best] if k == "absolute + Δ (headline)" else
                                         group_cv(best, d[c].to_numpy(float), y, grp)[0]) for k, c in sets.items()}}
    # ---- RUL task: remaining cycles to 80% SOH, for cells that reach it
    eol = cdf[cdf.ok & cdf.eol80_cycle.notna()].eol80_cycle.to_dict()
    r = d[d.battery_id.isin(eol)].copy()
    r["rul"] = r.battery_id.map(eol) - r.cycle_no
    r = r[r.rul >= 0].reset_index(drop=True)
    Xr, yr, gr = r[MODEL_COLS].to_numpy(float), r.rul.to_numpy(float), r.battery_id.to_numpy()
    res["rul"].update(n_cells=len(eol), n_rows=int(len(r)), eol_cycles={k: int(v) for k, v in eol.items()},
                      cells_ending_before_50=int(sum(v < 50 for v in eol.values())),
                      definition="RUL = discharge cycles remaining until SOH first reaches 80% of the cell's reference capacity")
    roof = {}
    for n in MODEL_NAMES:
        res["rul"]["cv"][n], roof[n] = group_cv(n, Xr, yr, gr, folds=5)
        log(f"[rul] {n:18s} R2={res['rul']['cv'][n]['R2']:.3f} RMSE={res['rul']['cv'][n]['RMSE']:.1f}")
    rb = min(MODEL_NAMES, key=lambda n: res["rul"]["cv"][n]["RMSE"])
    res["rul"]["best_model"] = rb
    Xb = r[["cycle_no", "ambient_c"]].to_numpy(float)
    res["rul"]["baseline_cycle_only"] = group_cv("Gradient Boosting", Xb, yr, gr)[0]
    res["rul"]["rows"] = [dict(b=q.battery_id, k=int(q.cycle_no), y=round(float(q.rul), 1), p=round(float(roof[rb][i]), 1)) for i, q in enumerate(r.itertuples())]
    # ---- final models + importances
    bundle = {"soh": {n: mk(n).fit(X, y) for n in MODEL_NAMES}, "rul": {n: mk(n).fit(Xr, yr) for n in MODEL_NAMES},
              "features": MODEL_COLS}
    res["importance"] = {}
    for n in ["XGBoost", "Gradient Boosting", "Random Forest"]:
        imp = bundle["soh"][n].regressor_.named_steps["m"].feature_importances_
        imp = 100 * imp / imp.max()
        res["importance"][n] = sorted(({"feature": f, "importance": float(v), "group": ("ICA" if f.replace("_d", "") in ICA_COLS else "charge/CV" if f.replace("_d", "") in CHARGE_COLS else "temperature") + (" Δ" if f in DELTA_COLS else "")}
                                       for f, v in zip(MODEL_COLS, imp)), key=lambda z: -z["importance"])
    res["sp1"] = sp1_summary()
    bundle["versions"] = dict(sklearn=sklearn.__version__, xgboost=xgboost.__version__, numpy=np.__version__)
    res["meta"]["train_seconds"], res["meta"]["versions"] = round(time.time() - t0, 1), bundle["versions"]
    joblib.dump(bundle, out_dir / BUNDLE)
    (out_dir / RES_JSON).write_text(json.dumps(res))
    log(f"[nasa done] soh best={best} rul best={rb} in {res['meta']['train_seconds']}s")
    return res, bundle


if __name__ == "__main__":
    run_nasa_training()
