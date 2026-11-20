"""Replicate the IEEE paper's EVALUATION PROTOCOL on the MIT-Stanford data we have.

Paper: stratified random split across cycle-life terciles (72 train / 52 test of 124 cells = 58% / 42%), features standardised on
the training set, 5-fold CV grid search on the training set (RMSE) with the stated hyperparameter ranges, raw cycle life as the
target, four models (XGBoost, Random Forest, SVR, Gradient Boosting), metrics R2 / RMSE / MAE / MAPE on the held-out test set.

Differences that remain (stated in the report): we have 138 cells (not the paper's 124) with 40 per-cycle summary features
(not 108 ICA features from raw voltage-capacity curves), so numbers are protocol-matched, not feature-matched."""
import json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from . import config as C
from .datasets import Registry
from .features import FEATURE_COLS
from .training import fit, make_model, metrics, predict

PAPER = {  # Table 2 of the paper (test set, n = 52 cells)
    "XGBoost": dict(R2=0.5067, RMSE=136.97, MAE=111.23, MAPE=16.13), "Random Forest": dict(R2=0.4019, RMSE=150.82, MAE=117.26, MAPE=17.24),
    "Gradient Boosting": dict(R2=0.3715, RMSE=154.61, MAE=126.04, MAPE=18.18), "SVR": dict(R2=0.1637, RMSE=178.34, MAE=144.55, MAPE=22.64)}
TEST_FRACTION = 52 / 124
GRIDS = {  # subsets of the paper's stated ranges (tree count 100-800, depth 3-10, lr 0.01-0.3; C 0.1-100, eps 0.01-1.0, RBF gamma)
    "XGBoost": (lambda: XGBRegressor(subsample=0.8, colsample_bytree=0.7, random_state=C.SEED, n_jobs=1),
                {"m__n_estimators": [100, 300, 600], "m__max_depth": [3, 6, 10], "m__learning_rate": [0.03, 0.1, 0.3]}),
    "Random Forest": (lambda: RandomForestRegressor(random_state=C.SEED, n_jobs=1),
                      {"m__n_estimators": [100, 300], "m__max_depth": [3, 6, 10]}),
    "Gradient Boosting": (lambda: GradientBoostingRegressor(subsample=0.8, random_state=C.SEED),
                          {"m__n_estimators": [100, 300], "m__max_depth": [3, 6], "m__learning_rate": [0.03, 0.1, 0.3]}),
    "SVR": (lambda: SVR(kernel="rbf"), {"m__C": [0.1, 1, 10, 100], "m__epsilon": [0.01, 0.1, 1.0], "m__gamma": ["scale", 0.01, 0.1]}),
}
IMPROVED = ["XGBoost", "Random Forest", "Gradient Boosting", "SVR", "Elastic Net"]


def split(y, seed):
    terc = pd.qcut(y, 3, labels=False)
    n_test = int(round(TEST_FRACTION * len(y)))
    return train_test_split(np.arange(len(y)), test_size=n_test, stratify=terc, random_state=seed)


def run_paper_protocol(out_dir=C.ART_DIR, seeds=tuple(range(42, 52)), log=print):
    t0 = time.time()
    full = Registry().feat["mit_all"]
    X, y = full[FEATURE_COLS].to_numpy(float), full.cycle_life.to_numpy(float)
    per = {m: [] for m in list(GRIDS) + ["improved:" + m for m in IMPROVED]}
    detail, params42 = {}, {}
    for s in seeds:
        tr, te = split(y, s)
        for name, (mk, grid) in GRIDS.items():
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", mk())])
            gs = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=s), scoring="neg_root_mean_squared_error", n_jobs=1).fit(X[tr], y[tr])
            p = gs.predict(X[te])
            per[name].append(metrics(y[te], p))
            if s == seeds[0]:
                params42[name] = {k.replace("m__", ""): v for k, v in gs.best_params_.items()}
                detail[name] = dict(ids=full.battery_id.iloc[te].tolist(), actual=y[te].tolist(), pred=[float(v) for v in p])
        for name in IMPROVED:                     # same split, our pipeline: log target + fixed regularised hyperparameters
            p = predict(fit(make_model(name), X[tr], y[tr]), X[te])
            per["improved:" + name].append(metrics(y[te], p))
            if s == seeds[0]:
                detail["improved:" + name] = dict(ids=full.battery_id.iloc[te].tolist(), actual=y[te].tolist(), pred=[float(v) for v in p])
        log(f"[paper] seed {s} done ({time.time() - t0:.0f}s)  XGB R2={per['XGBoost'][-1]['R2']:.2f}  improved-XGB R2={per['improved:XGBoost'][-1]['R2']:.2f}")
    summ = {}
    for k, runs in per.items():
        summ[k] = {"single": runs[0], "mean": {m: float(np.mean([r[m] for r in runs])) for m in runs[0]},
                   "std": {m: float(np.std([r[m] for r in runs])) for m in runs[0]}, "per_seed_r2": [round(r["R2"], 4) for r in runs]}
    res = {"paper": PAPER, "seeds": list(seeds), "n_cells": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)), "n_features": len(FEATURE_COLS),
           "results": summ, "best_params_first_seed": params42, "first_seed_predictions": detail, "seconds": round(time.time() - t0),
           "caveats": ["138 cells (paper: 124) and 40 summary features (paper: 108 ICA features): protocol-matched, not feature-matched.",
                       f"{len(seeds)} random splits are reported because a single split is a noisy draw; the first seed ({seeds[0]}) is the single-split number.",
                       "Grids are subsets of the paper's stated hyperparameter ranges (compute limits)."]}
    (out_dir / "paper_results.json").write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    run_paper_protocol()
