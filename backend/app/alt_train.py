"""Train and evaluate on the NASA/UCF accelerated-life packs: SOH from charge-side signals (reference discharges give the labels),
remaining life to 80% capacity, a second-life transfer test, and life vs load. All validation is leave-pack-out."""
import json, time
import numpy as np
import pandas as pd
import joblib, sklearn, xgboost
from . import alt
from . import config as C
from .nasa_train import agg, group_cv, met, mk
from .training import MODEL_NAMES

CYC_CSV, CURVES, RES_JSON, BUNDLE = "alt_cycles.csv", "alt_ica_curves.json", "alt_results.json", "alt_models.joblib"
ICA_F = alt.ICA_COLS
CHG_F = ["ch_dur_s", "ch_v_start", "ch_v_q25", "ch_v_q50", "ch_v_q75", "ch_t_mean", "ch_t_max", "rest_v", "rest_s", "v_start", "ir_proxy",
         "t_mean", "t_max", "load_a"]
CHG_D = [f + "_d" for f in CHG_F if f != "load_a"]
ICA_D = [f + "_d" for f in ICA_F]
EOL = 0.80
SOH_MIN = 50.0           # reference-discharge SOH below this is a failure collapse, not gradual ageing


def build(out_dir=C.ART_DIR):
    d = pd.read_csv(out_dir / CYC_CSV, dtype={"pack": str}).sort_values(["pack", "cycle"]).reset_index(drop=True)
    d["ir_proxy"] = (d.rest_v - d.v_start) / d.i_mean
    parts, packs = [], []
    for p, g in d.groupby("pack", sort=True):
        g = g.copy()
        grp, kind, loads = alt.LOADS[p]
        reg = np.where(g.mission == 1, g.i_mean, np.nan)
        g["load_a"] = pd.Series(reg, index=g.index).rolling(5, min_periods=1).median().ffill().bfill()
        ref = g[g.mission == 0]
        c0r = float(ref.cap_ah.iloc[:2].median()) if len(ref) else np.nan
        g["soh"] = np.where(g.mission == 0, g.cap_ah / c0r * 100, np.nan)
        for f in CHG_F:                       # start-of-life change, baseline = pack's first 5 discharge cycles
            if f != "load_a":
                g[f + "_d"] = g[f] - g[f].iloc[:5].median()
        for f in ICA_F:                       # baseline = first 2 reference discharges
            g[f + "_d"] = g[f] - ref[f].iloc[:2].median() if len(ref) else np.nan
        c0 = float(g.cap_ah.iloc[:3].median())
        sm = g.cap_ah.rolling(5, center=True, min_periods=2).median().to_numpy()
        hit = np.where(sm <= EOL * c0)[0]
        eol = int(g.cycle.iloc[hit[0]]) if len(hit) else None
        g["eol_cycle"] = float(eol) if eol is not None else np.nan
        n_ref_ok = int(((g.soh >= SOH_MIN) & g.soh.notna()).sum())
        packs.append(dict(pack=p, group=grp, kind=kind, loads=loads, load_label="/".join(f"{x:g}" for x in loads), cycles=int(len(g)),
                          ref_cycles=int(len(ref)), ref_ok=n_ref_ok, c0_ah=round(c0, 3), eol80_cycle=eol, censored=eol is None,
                          cap_min_ratio=round(float(np.nanmin(sm) / c0), 3)))
        parts.append(g)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(packs).set_index("pack")


def life_vs_load(pk):
    """Constant/variable-load regular packs that reached 80%: log(life) = a + b*log(load); leave-one-pack-out."""
    r = pk[(pk.group == "regular") & pk.eol80_cycle.notna()].copy()
    x, y = np.log(r.loads.map(lambda v: v[0]).astype(float)), np.log(r.eol80_cycle.astype(float))
    b, a = np.polyfit(x, y, 1)
    loo = []
    for i in range(len(r)):
        m = np.arange(len(r)) != i
        bb, aa = np.polyfit(x[m], y[m], 1)
        loo.append(float(np.exp(aa + bb * x.iloc[i])))
    act = r.eol80_cycle.astype(float).to_numpy()
    return dict(n=int(len(r)), slope=float(b), intercept=float(a), loo_mape=float(np.mean(np.abs(np.array(loo) - act) / act) * 100),
                loo_rmse=float(np.sqrt(np.mean((np.array(loo) - act) ** 2))),
                points=[dict(pack=i, load=float(r.loads[i][0]), life=int(r.eol80_cycle[i]), loo=float(l)) for i, l in zip(r.index, loo)])


def run_alt_training(out_dir=C.ART_DIR, log=print):
    t0 = time.time()
    d, pk = build(out_dir)
    # ---- SOH task: reference discharges (clean 2.5 A labels), packs with enough usable references
    s = d[(d.mission == 0) & d.soh.notna() & (d.soh >= SOH_MIN)].copy()
    ok_packs = pk[pk.ref_ok >= 4].index
    s = s[s.pack.isin(ok_packs)].reset_index(drop=True)
    y, grp = s.soh.to_numpy(float), s.pack.to_numpy()
    HEAD = CHG_D + ["load_a"]               # headline: changes from the pack's own start (+ current load level); chosen by CV from the sets below
    res = {"meta": {"packs": int(len(pk)), "cycles": int(len(d)), "reference_cycles": int((d.mission == 0).sum()), "soh_rows": int(len(s)),
                    "soh_packs": int(s.pack.nunique()), "cv_folds": C.CV_FOLDS, "cv_repeats": C.CV_REPEATS, "eol_fraction": EOL,
                    "ica_window_v": [5.9, 8.0], "source": "Fricke, Nascimento, Viana (2023), NASA / UCF accelerated life testing dataset (MIT licence)"},
           "packs": json.loads(pk.reset_index().assign(loads=lambda x: x.loads.map(list)).to_json(orient="records")),
           "soh": {"cv": {}}, "rul": {"cv": {}}}
    oof = {}
    for n in MODEL_NAMES:
        res["soh"]["cv"][n], oof[n] = group_cv(n, s[HEAD].to_numpy(float), y, grp)
        log(f"[alt soh] {n:18s} R2={res['soh']['cv'][n]['R2']:.3f} RMSE={res['soh']['cv'][n]['RMSE']:.2f}")
    best = min(MODEL_NAMES, key=lambda n: res["soh"]["cv"][n]["RMSE"])
    res["soh"]["best_model"] = best
    res["soh"]["rows"] = [dict(b=r.pack, k=int(r.cycle), y=round(float(r.soh), 2), p=round(float(oof[best][i]), 2)) for i, r in enumerate(s.itertuples())]
    res["soh"]["oof"] = {n: [round(float(v), 2) for v in oof[n]] for n in MODEL_NAMES}
    res["soh"]["true"] = [round(float(v), 2) for v in y]
    res["soh"]["per_pack"] = [dict(pack=b, n=int(len(g)), **met(g.y.to_numpy(), g.p.to_numpy())) for b, g in pd.DataFrame(res["soh"]["rows"]).groupby("b")]
    sets = {"charge-side Δ (+load) (headline)": HEAD, "charge-side abs": CHG_F, "charge-side abs + Δ": CHG_F + CHG_D,
            "reference-discharge ICA abs": ICA_F, "reference-discharge ICA Δ": ICA_D,
            "ICA + charge-side (needs ref. discharge)": ICA_F + ICA_D + CHG_F + CHG_D, "cycle number + load only": ["cycle", "load_a"]}
    res["soh"]["ablation"] = {best: {k: (res["soh"]["cv"][best] if k.endswith("(headline)") else group_cv(best, s[c].to_numpy(float), y, grp)[0]) for k, c in sets.items()}}
    # ---- second-life transfer: train on all other packs, test on the three second-life packs
    tr, te = s[s.pack.map(pk.group) != "second_life"], s[s.pack.map(pk.group) == "second_life"]
    if len(te) and len(tr):
        pr = mk(best).fit(tr[HEAD].to_numpy(float), tr.soh.to_numpy(float)).predict(te[HEAD].to_numpy(float))
        res["soh"]["second_life_transfer"] = {**met(te.soh.to_numpy(float), pr), "n_rows": int(len(te)), "n_packs": int(te.pack.nunique()),
                                              "note": "Trained on first-life packs only; scored on second-life packs it never saw."}
    # ---- RUL task: cycles remaining to 80% capacity, packs that reach it (all cycles up to that point)
    r = d[d.eol_cycle.notna() & (d.cycle <= d.eol_cycle)].copy()
    r["rul"] = r.eol_cycle - r.cycle
    r = r.reset_index(drop=True)
    RH = HEAD
    yr, gr = r.rul.to_numpy(float), r.pack.to_numpy()
    res["rul"].update(n_packs=int(r.pack.nunique()), n_rows=int(len(r)), eol_cycles={p: int(v) for p, v in pk.eol80_cycle.dropna().items()},
                      definition="RUL = discharge cycles remaining until capacity first reaches 80% of the pack's initial capacity (5-cycle median)")
    roof = {}
    for n in MODEL_NAMES:
        res["rul"]["cv"][n], roof[n] = group_cv(n, r[RH].to_numpy(float), yr, gr)
        log(f"[alt rul] {n:18s} R2={res['rul']['cv'][n]['R2']:.3f} RMSE={res['rul']['cv'][n]['RMSE']:.1f}")
    rb = min(MODEL_NAMES, key=lambda n: res["rul"]["cv"][n]["RMSE"])
    res["rul"]["best_model"] = rb
    rs = {"cycle + load only (baseline)": ["cycle", "load_a"], "charge-side abs": CHG_F, "charge-side abs + Δ": CHG_F + CHG_D,
          "charge-side Δ + cycle": HEAD + ["cycle"], "charge-side Δ (+load) (headline)": RH}
    res["rul"]["ablation"] = {rb: {k: (res["rul"]["cv"][rb] if k.endswith("(headline)") else group_cv(rb, r[c].to_numpy(float), yr, gr)[0]) for k, c in rs.items()}}
    res["rul"]["rows"] = [dict(b=q.pack, k=int(q.cycle), y=int(q.rul), p=round(float(roof[rb][i]), 1)) for i, q in enumerate(r.itertuples())]
    res["rul"]["per_pack"] = [dict(pack=b, n=int(len(g)), **met(g.y.to_numpy(float), g.p.to_numpy(float))) for b, g in pd.DataFrame(res["rul"]["rows"]).groupby("b")]
    res["life_vs_load"] = life_vs_load(pk)
    # ---- final models + importances
    bundle = {"soh": {n: mk(n).fit(s[HEAD].to_numpy(float), y) for n in MODEL_NAMES},
              "rul": {n: mk(n).fit(r[RH].to_numpy(float), yr) for n in MODEL_NAMES}, "soh_features": HEAD, "rul_features": RH}
    res["importance"] = {}
    for key, feats in (("soh", HEAD), ("rul", RH)):
        n = best if key == "soh" else rb
        mdl = bundle[key][n].regressor_.named_steps["m"]
        if hasattr(mdl, "feature_importances_"):
            imp = 100 * mdl.feature_importances_ / mdl.feature_importances_.max()
            res["importance"][key] = dict(model=n, top=sorted(({"feature": f, "importance": float(v)} for f, v in zip(feats, imp)), key=lambda z: -z["importance"])[:12])
    curves = json.loads((out_dir / CURVES).read_text())
    res["ica"] = {"v": alt.VPLOT.tolist(), "curves": {p: v for p, v in curves.items()}}
    res["capacity"] = {p: [dict(k=int(a), c=round(float(b), 3), m=int(c)) for a, b, c in zip(g.cycle, g.cap_ah, g.mission)] for p, g in d.groupby("pack")}
    bundle["versions"] = dict(sklearn=sklearn.__version__, xgboost=xgboost.__version__, numpy=np.__version__)
    res["meta"]["train_seconds"], res["meta"]["versions"] = round(time.time() - t0, 1), bundle["versions"]
    joblib.dump(bundle, out_dir / BUNDLE)
    (out_dir / RES_JSON).write_text(json.dumps(res))
    log(f"[alt done] soh best={best} rul best={rb} in {res['meta']['train_seconds']}s")
    return res, bundle


if __name__ == "__main__":
    run_alt_training()
