"""Training + honest evaluation: repeated stratified CV, per-dataset CV, leave-one-batch-out,
window sweep, protocol ablation, quantile prediction intervals."""
import json, sys, time, platform
import numpy as np
import pandas as pd
import joblib, sklearn, xgboost
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from . import config as C
from .datasets import Registry
from .features import CAT_OF, DEG_COLS, FEATURE_COLS, feature_table

MODEL_NAMES = ["Elastic Net", "XGBoost", "Random Forest", "Gradient Boosting", "SVR"]


def make_model(name, kind="mean"):
    s = C.SEED
    if kind in ("lo", "hi"):
        est = GradientBoostingRegressor(loss="quantile", alpha=0.1 if kind == "lo" else 0.9, n_estimators=150,
                                        max_depth=2, learning_rate=0.06, subsample=0.8, min_samples_leaf=3,
                                        random_state=s)
    else:
        est = {
            "Elastic Net": ElasticNetCV(l1_ratio=[.5, .9, 1], cv=3, max_iter=20000),
            "XGBoost": XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.7, min_child_weight=2, reg_lambda=2, random_state=s, n_jobs=1),
            "Random Forest": RandomForestRegressor(150, max_depth=8, min_samples_leaf=2, max_features=0.5,
                                                   random_state=s, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                           subsample=0.8, min_samples_leaf=3, random_state=s),
            "SVR": SVR(C=3, epsilon=0.05, gamma="scale"),
        }[name]
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -5, "a_max": 5})), ("m", est)])


def fit(pipe, X, y):
    return pipe.fit(X, np.log(y))          # models predict log(cycle life)


def predict(pipe, X):
    return np.exp(np.clip(pipe.predict(X), np.log(50), np.log(5000)))   # plausible 50-5000 cycles


def metrics(y, p):
    return dict(R2=float(r2_score(y, p)), RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
                MAE=float(mean_absolute_error(y, p)), MAPE=float(np.mean(np.abs(y - p) / y) * 100))


def folds(y, seed):
    bins = pd.qcut(y, min(5, max(2, len(y) // 10)), labels=False, duplicates="drop")
    return StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), bins)


def cv(name, X, y, kinds=("mean",)):
    """Repeated stratified CV. Returns metrics mean/std over repeats + repeat-0 out-of-fold preds."""
    runs, first = [], {}
    for r in range(C.CV_REPEATS):
        oof = {k: np.zeros(len(y)) for k in kinds}
        for tr, te in folds(y, C.SEED + r):
            for k in kinds:
                oof[k][te] = predict(fit(make_model(name, k), X[tr], y[tr]), X[te])
        runs.append(metrics(y, oof["mean"]))
        if r == 0:
            first = oof
    out = {}
    for m in runs[0]:
        v = [x[m] for x in runs]
        out[m], out[m + "_std"] = float(np.mean(v)), float(np.std(v))
    return out, first


def agg(runs):
    out = {}
    for m in runs[0]:
        v = [x[m] for x in runs]
        out[m], out[m + "_std"] = float(np.mean(v)), float(np.std(v))
    return out


def label_of(c1, q1, c2):
    if any(pd.isna(v) for v in (c1, q1, c2)):
        return "unknown"
    return f"{c1:g}C({q1:g}%)-{c2:g}C"


def protocol_cv(name, X, y, groups, cols=None):
    """Leave-protocol-out CV: whole charging protocols are held out together (no replicate leakage)."""
    Xa = X if cols is None else X[:, cols]
    ug, runs = np.unique(groups), []
    for r in range(C.CV_REPEATS):
        parts = np.array_split(np.random.default_rng(C.SEED + r).permutation(ug), C.CV_FOLDS)
        oof = np.zeros(len(y))
        for part in parts:
            te = np.isin(groups, part)
            oof[te] = predict(fit(make_model(name), Xa[~te], y[~te]), Xa[te])
        runs.append(metrics(y, oof))
    return agg(runs)


def nn(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def summarize(tbl):
    return tbl[FEATURE_COLS].to_numpy(float), tbl["cycle_life"].to_numpy(float)


def run_training(reg: Registry, out_dir=C.ART_DIR, log=print):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"meta": {"early_cycles": C.EARLY_CYCLES, "n_features": len(FEATURE_COLS), "cv_folds": C.CV_FOLDS,
                    "cv_repeats": C.CV_REPEATS, "target": "log(cycle life)",
                    "tiers": {"good_above": C.TIER_GOOD, "poor_below": C.TIER_POOR}}}
    # ---- 1. per-dataset repeated CV, all models
    res["cv"], oof_all = {}, {}
    for did in [d for d in reg.meta if reg.meta[d]["builtin"]]:
        X, y = summarize(reg.feat[did])
        res["cv"][did] = {}
        for n in MODEL_NAMES:
            m, oof = cv(n, X, y, kinds=("mean", "lo", "hi") if (did == "mit_all" and n == "Gradient Boosting") else ("mean",))
            res["cv"][did][n] = m
            if did == "mit_all":
                oof_all[n] = oof
            log(f"[cv] {did:8s} {n:18s} R2={m['R2']:.3f}±{m['R2_std']:.3f} RMSE={m['RMSE']:.1f}")
    best = min(MODEL_NAMES, key=lambda n: res["cv"]["mit_all"][n]["RMSE"])
    res["meta"]["best_model"] = best
    # ---- 2. out-of-fold predictions (repeat 0) for the combined set
    full = reg.feat["mit_all"]
    ids, y = full.battery_id.tolist(), full.cycle_life.to_numpy(float)
    res["oof"] = {n: {"ids": ids, "actual": y.tolist(), "pred": oof_all[n]["mean"].tolist()} for n in MODEL_NAMES}
    # intervals come from quantile GB OOF predictions; centre = best model
    lo = np.minimum(oof_all["Gradient Boosting"]["lo"], oof_all[best]["mean"])
    hi = np.maximum(oof_all["Gradient Boosting"]["hi"], oof_all[best]["mean"])
    res["interval"] = {"nominal": 0.8, "empirical_coverage": float(np.mean((y >= lo) & (y <= hi))),
                       "mean_width": float(np.mean(hi - lo))}
    cells = []
    for i, b in enumerate(ids):
        pr = float(oof_all[best]["mean"][i])
        cells.append(dict(battery_id=b, batch=full.batch[i], actual_cycle_life=float(y[i]), predicted_cycle_life=pr,
                          lo=float(lo[i]), hi=float(hi[i]), actual_rul=float(y[i] - C.EARLY_CYCLES),
                          predicted_rul=pr - C.EARLY_CYCLES, tier=C.tier(pr - C.EARLY_CYCLES),
                          actual_tier=C.tier(y[i] - C.EARLY_CYCLES), source="out-of-fold",
                          c1=nn(full.c1_rate[i]), q1=nn(full.q1_switch_pct[i]), c2=nn(full.c2_rate[i]),
                          protocol=label_of(full.c1_rate[i], full.q1_switch_pct[i], full.c2_rate[i])))
    res["cells"] = cells
    res["tier_accuracy"] = float(np.mean([c["tier"] == c["actual_tier"] for c in cells]))
    # ---- 3. leave-one-batch-out (true cross-dataset generalisation)
    res["cross_batch"] = {}
    for hold in ["1", "2", "3"]:
        tr, te = full[full.batch != hold], full[full.batch == hold]
        Xtr, ytr = summarize(tr); Xte, yte = summarize(te)
        res["cross_batch"][f"hold_out_b{hold}"] = {
            "n_train": len(tr), "n_test": len(te),
            "models": {n: metrics(yte, predict(fit(make_model(n), Xtr, ytr), Xte)) for n in MODEL_NAMES}}
        log(f"[lobo] hold b{hold}: " + " ".join(f"{n}={v['R2']:.2f}" for n, v in res['cross_batch'][f'hold_out_b{hold}']['models'].items()))
    # ---- 3b. charging-protocol analysis: leave-protocol-out CV
    Xf, yf = summarize(full)
    labels = np.array([label_of(a, b_, c_) for a, b_, c_ in zip(full.c1_rate, full.q1_switch_pct, full.c2_rate)])
    res["protocol_cv"] = {n: protocol_cv(n, Xf, yf, labels) for n in MODEL_NAMES}
    deg = [FEATURE_COLS.index(c) for c in DEG_COLS]
    res["protocol_ablation"] = {best: {"with_protocol": res["protocol_cv"][best],
                                       "without_protocol": protocol_cv(best, Xf, yf, labels, cols=deg)}}
    tab = []
    for lab in np.unique(labels):
        sub = full[labels == lab]
        tab.append(dict(label=lab, n=int(len(sub)), life_mean=float(sub.cycle_life.mean()), life_min=float(sub.cycle_life.min()),
                        life_max=float(sub.cycle_life.max()), batches=sorted(set(sub.batch)),
                        c1=nn(sub.c1_rate.iloc[0]), q1=nn(sub.q1_switch_pct.iloc[0]), c2=nn(sub.c2_rate.iloc[0])))
    res["protocols"] = sorted(tab, key=lambda d: (-d["n"], d["label"]))
    log("[protocol-cv] " + " ".join(f"{n}={v['R2']:.2f}" for n, v in res["protocol_cv"].items()))
    # ---- 4. window sweep (same full feature set, same CV)
    res["window"] = {}
    for n in dict.fromkeys(["Elastic Net", best]):
        res["window"][n] = {}
        for w in [10, 20, 30, 40, 50]:
            t = feature_table(reg.raw, window=w, min_rows=int(0.8 * w))
            t = t[t.cycle_life.notna()]
            Xw, yw = summarize(t)
            res["window"][n][str(w)] = {**cv(n, Xw, yw)[0], "n": len(t)}
    # ---- 5. protocol ablation: do charging-protocol features carry the signal?
    X, y = summarize(full)
    deg_idx = [FEATURE_COLS.index(c) for c in DEG_COLS]
    res["ablation"] = {best: {"with_protocol": res["cv"]["mit_all"][best],
                              "without_protocol": cv(best, X[:, deg_idx], y)[0]}}
    # ---- 6. final models (fit on everything) + importances
    bundle = {"models": {}, "lo": None, "hi": None}
    for n in MODEL_NAMES:
        bundle["models"][n] = fit(make_model(n), X, y)
    bundle["lo"], bundle["hi"] = (fit(make_model("Gradient Boosting", k), X, y) for k in ("lo", "hi"))
    res["importance"] = {}
    for n in ["XGBoost", "Random Forest", "Gradient Boosting"]:
        imp = bundle["models"][n].named_steps["m"].feature_importances_
        imp = 100 * imp / imp.max()
        res["importance"][n] = sorted(({"feature": f, "importance": float(v), "category": CAT_OF[f]}
                                       for f, v in zip(FEATURE_COLS, imp)), key=lambda d: -d["importance"])
    versions = dict(python=platform.python_version(), sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                    numpy=np.__version__, pandas=pd.__version__)
    bundle["versions"] = versions
    res["meta"]["versions"], res["meta"]["train_seconds"] = versions, round(time.time() - t0, 1)
    joblib.dump(bundle, out_dir / "models.joblib")
    (out_dir / "results.json").write_text(json.dumps(res))
    log(f"[done] best={best} in {res['meta']['train_seconds']}s")
    return res, bundle


if __name__ == "__main__":
    run_training(Registry())
