#!/usr/bin/env python3
"""Generate results/ (figures, CSV tables, RESULTS.md) from the trained-model result files in backend/artifacts/.
Usage (from the project root):  python tools/make_report.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART, OUT = ROOT / "backend" / "artifacts", ROOT / "results"
FIG, TAB = OUT / "figures", OUT / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)
J = lambda n: json.loads((ART / n).read_text())
M, N, A, P = J("results.json"), J("nasa_results.json"), J("alt_results.json"), J("paper_results.json")
NI, SP = J("nasa_ica_curves.json"), N["sp1"]
FOUR = ["XGBoost", "Random Forest", "Gradient Boosting", "SVR"]
C = ["#2b8cbe", "#31a354", "#e6550d", "#756bb1", "#636363"]
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def save(name):
    plt.tight_layout(); plt.savefig(FIG / name); plt.close()


def scatter(ax, x, y, lim, xl, yl):
    ax.scatter(x, y, s=8, alpha=.5, color=C[0]); ax.plot(lim, lim, "--", color="grey"); ax.set_xlabel(xl); ax.set_ylabel(yl)


# ---- 1. paper replication
r = P["results"]; x = np.arange(4); w = .27
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(x - w, [P["paper"][k]["R2"] for k in FOUR], w, label="paper (Table 2)", color="#969696")
ax.bar(x, [r[k]["mean"]["R2"] for k in FOUR], w, yerr=[r[k]["std"]["R2"] for k in FOUR], label="ours, paper protocol (mean of %d splits)" % len(P["seeds"]), color=C[0], capsize=3)
ax.bar(x + w, [r["improved:" + k]["mean"]["R2"] for k in FOUR], w, yerr=[r["improved:" + k]["std"]["R2"] for k in FOUR], label="ours, improved pipeline", color=C[1], capsize=3)
ax.set_xticks(x, FOUR); ax.set_ylabel("test R²"); ax.set_ylim(0, 0.95); ax.legend(fontsize=7, loc="upper right"); ax.set_title("Paper protocol re-run on MIT-Stanford summary features")
save("01_paper_replication_r2.png")
fig, ax = plt.subplots(figsize=(7, 3.4))
for i, k in enumerate(FOUR): ax.plot(range(1, len(P["seeds"]) + 1), r[k]["per_seed_r2"], "o-", color=C[i], label=k, ms=3)
ax.axhline(P["paper"]["XGBoost"]["R2"], ls="--", color="grey", label="paper XGBoost 0.507"); ax.set_xlabel("random split"); ax.set_ylabel("test R²"); ax.legend(fontsize=7, ncol=3)
save("02_paper_split_variation.png")
d = P["first_seed_predictions"]["XGBoost"]; fig, ax = plt.subplots(figsize=(4.4, 4)); lim = [0, max(d["actual"] + d["pred"]) * 1.05]
scatter(ax, d["actual"], d["pred"], lim, "actual cycle life", "predicted"); ax.set_title("XGBoost, paper protocol, first split (n=%d)" % len(d["actual"])); save("03_paper_xgb_scatter.png")
# ---- 2. MIT generalisation
best = M["meta"]["best_model"]; names = list(M["cv"]["mit_all"])
rnd = [M["cv"]["mit_all"][n]["R2"] for n in names]; pcv = [M["protocol_cv"][n]["R2"] for n in names]
bcv = [np.mean([M["cross_batch"][h]["models"][n]["R2"] for h in M["cross_batch"]]) for n in names]
fig, ax = plt.subplots(figsize=(7.5, 3.6)); x = np.arange(len(names))
ax.bar(x - w, rnd, w, label="random-split CV", color=C[0]); ax.bar(x, pcv, w, label="leave-protocol-out", color=C[2]); ax.bar(x + w, bcv, w, label="leave-batch-out (mean)", color=C[3])
ax.axhline(0, color="k", lw=.6); ax.set_xticks(x, names); ax.set_ylabel("R²"); ax.legend(fontsize=7); ax.set_title("MIT-Stanford: accuracy falls as validation gets stricter")
save("04_mit_generalization.png")
o = M["oof"][best]; fig, ax = plt.subplots(figsize=(4.4, 4)); scatter(ax, o["actual"], o["pred"], [0, 2000], "actual cycle life", "out-of-fold prediction"); ax.set_title("MIT-Stanford, out-of-fold (%s)" % best); save("05_mit_oof_scatter.png")
# ---- 3. NASA PCoE
nb = N["soh"]["best_model"]; fig, ax = plt.subplots(figsize=(4.4, 4)); scatter(ax, N["soh"]["true"], N["soh"]["oof"][nb], [50, 110], "measured SOH %", "predicted SOH % (out-of-fold)"); ax.set_title("NASA PCoE SOH, leave-battery-out (%s)" % nb); save("06_nasa_soh_scatter.png")
ab = N["soh"]["ablation"][nb]; fig, ax = plt.subplots(figsize=(7, 3.6)); ax.barh(list(ab)[::-1], [v["R2"] for v in ab.values()][::-1], color=["#31a354" if "headline" in k else C[0] for k in list(ab)[::-1]])
ax.set_xlabel("R²"); ax.set_title("NASA PCoE: what carries the signal (%s)" % nb); save("07_nasa_ablation.png")
fig, ax = plt.subplots(figsize=(6, 3.4))
for c in NI["cells"]["B0005"][::2]: ax.plot(NI["v"], c["ic"], label="cycle %d" % c["k"])
ax.set_xlabel("voltage (V)"); ax.set_ylabel("dQ/dV (Ah/V)"); ax.set_title("NASA PCoE B0005: ICA of the charge curve"); ax.legend(fontsize=7); save("08_nasa_ica_B0005.png")
# ---- 4. accelerated-life packs
V = A["ica"]["v"]; cu = A["ica"]["curves"]["01"]; fig, ax = plt.subplots(figsize=(6, 3.4))
for i in np.linspace(0, len(cu) - 1, 5).astype(int): ax.plot(V, cu[i]["ic"], label="cycle %d" % cu[i]["k"])
ax.set_xlabel("pack voltage (V)"); ax.set_ylabel("|dQ/dV| (Ah/V)"); ax.set_title("Pack 01: ICA of 2.5 A reference discharges over its life"); ax.legend(fontsize=7); save("09_alt_ica_aging_pack01.png")
ab_ = A["soh"]["best_model"]; fig, ax = plt.subplots(figsize=(4.4, 4)); scatter(ax, A["soh"]["true"], A["soh"]["oof"][ab_], [50, 110], "measured SOH %", "predicted SOH % (out-of-fold)"); ax.set_title("Packs: SOH, leave-pack-out (%s)" % ab_); save("10_alt_soh_scatter.png")
rr = A["rul"]["rows"]; fig, ax = plt.subplots(figsize=(4.4, 4)); mx = max(q["y"] for q in rr); scatter(ax, [q["y"] for q in rr], [q["p"] for q in rr], [0, mx], "actual remaining cycles", "predicted"); ax.set_title("Packs: remaining life (%s)" % A["rul"]["best_model"]); save("11_alt_rul_scatter.png")
lv = A["life_vs_load"]; fig, ax = plt.subplots(figsize=(4.6, 3.6)); ax.loglog([p["load"] for p in lv["points"]], [p["life"] for p in lv["points"]], "o", color=C[0])
xs = np.linspace(min(p["load"] for p in lv["points"]) * .9, max(p["load"] for p in lv["points"]) * 1.1, 50); ax.plot(xs, np.exp(lv["intercept"] + lv["slope"] * np.log(xs)), color=C[2])
ax.set_xlabel("load (A)"); ax.set_ylabel("cycles to 80% capacity"); ax.set_title("Life vs load (slope %.2f, n=%d)" % (lv["slope"], lv["n"])); save("12_alt_life_vs_load.png")
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2)); pr = SP["profile"]; axs[0].plot(pr["hours"], pr["v"]); axs[0].set_xlabel("test time (h)"); axs[0].set_ylabel("V"); axs[0].set_title("SP20-1 reference test")
axs[1].plot(SP["ica"]["v"], SP["ica"]["ic"], color=C[1]); axs[1].set_xlabel("voltage (V)"); axs[1].set_ylabel("dQ/dV (Ah/V)"); axs[1].set_title("SP20-1: ICA of the 1 A charge (initial capacity %.2f Ah)" % SP["initial_capacity_ah"], fontsize=8); save("13_sp20_reference.png")

# ---- tables
def tab(name, rows): pd.DataFrame(rows).to_csv(TAB / name, index=False)
tab("paper_vs_ours.csv", [{"model": k, **{f"paper_{m}": P["paper"][k][m] for m in ("R2", "RMSE", "MAE", "MAPE")},
      **{f"ours_paper_protocol_mean_{m}": r[k]["mean"][m] for m in ("R2", "RMSE", "MAE", "MAPE")}, "ours_paper_protocol_R2_sd": r[k]["std"]["R2"],
      **{f"ours_improved_mean_{m}": r["improved:" + k]["mean"][m] for m in ("R2", "RMSE", "MAE", "MAPE")}} for k in FOUR])
tab("mit_stanford_models.csv", [{"model": n, "random_cv_R2": M["cv"]["mit_all"][n]["R2"], "random_cv_RMSE": M["cv"]["mit_all"][n]["RMSE"], "leave_protocol_out_R2": M["protocol_cv"][n]["R2"],
      **{f"leave_{h.replace('hold_out_', '')}_out_R2": M["cross_batch"][h]["models"][n]["R2"] for h in M["cross_batch"]}} for n in names])
for nm, d_ in (("nasa_pcoe_soh_models.csv", N["soh"]["cv"]), ("alt_packs_soh_models.csv", A["soh"]["cv"]), ("alt_packs_rul_models.csv", A["rul"]["cv"])):
    tab(nm, [{"model": k, **{m: v[m] for m in ("R2", "RMSE", "MAE")}} for k, v in d_.items()])
tab("nasa_pcoe_cell_quality_gate.csv", N["cells"]); tab("alt_pack_summary.csv", [{**p, "loads": "/".join(map(str, p["loads"]))} for p in A["packs"]])

# ---- RESULTS.md
f2 = lambda v: f"{v:.2f}"; f0 = lambda v: f"{v:,.0f}"
pn = P["n_test"]; nS = len(P["seeds"]); rk = sorted(FOUR, key=lambda k: -r[k]["mean"]["R2"])
nsc, ns = N["soh"]["cv"][nb], N["soh"]["cv"][nb]; ac, rc = A["soh"]["cv"][ab_], A["rul"]["cv"][A["rul"]["best_model"]]; rb = A["rul"]["best_model"]
base = A["rul"]["ablation"][rb]["cycle + load only (baseline)"]; tr = A["soh"].get("second_life_transfer", {})
md = f"""# Results: early-cycle battery life prediction, ICA + machine learning

Generated from the trained models in `backend/artifacts/` by `tools/make_report.py`. Every number is cross-validated or held-out; none is a training-set score.

## 1. Replicating the paper's protocol (MIT-Stanford data)

The paper's evaluation protocol was re-run: stratified split across cycle-life terciles ({P['n_train']} train / {pn} test, the paper's 58%/42%), standardised inputs, 5-fold grid search on the training set, raw cycle life as the target, four models, R2 / RMSE / MAE / MAPE. Run over {nS} random splits.

| Model | Paper R2 | Ours R2 (paper protocol, mean ± sd) | Ours R2 (improved, mean ± sd) | Paper MAPE | Ours MAPE (paper protocol) | Ours MAPE (improved) |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {k} | {P['paper'][k]['R2']:.3f} | {r[k]['mean']['R2']:.3f} ± {r[k]['std']['R2']:.3f} | {r['improved:'+k]['mean']['R2']:.3f} ± {r['improved:'+k]['std']['R2']:.3f} | {P['paper'][k]['MAPE']:.1f}% | {r[k]['mean']['MAPE']:.1f}% | {r['improved:'+k]['mean']['MAPE']:.1f}% |" for k in FOUR) + f"""

![paper replication](figures/01_paper_replication_r2.png)
![split variation](figures/02_paper_split_variation.png)

**Reading it honestly**
- The model ranking by mean R2 here is {' > '.join(rk)}; the paper's is XGBoost > Random Forest > Gradient Boosting > SVR. XGBoost best and SVR worst reproduce; the middle two swap within split noise.
- R2 is higher than the paper's for all four models, but this is **not a like-for-like comparison**: we have {P['n_cells']} cells (paper: 124) and {P['n_features']} per-cycle summary features (paper: 108 ICA features from raw voltage-capacity curves). The numbers are protocol-matched, not feature-matched.
- A single split is a noisy draw (XGBoost R2 ranges {min(r['XGBoost']['per_seed_r2']):.2f} to {max(r['XGBoost']['per_seed_r2']):.2f} across splits), so the paper's 0.507 and our first-split {r['XGBoost']['single']['R2']:.2f} both carry real uncertainty.
- Our improved pipeline (log target, fixed regularised hyperparameters) lowers MAPE for {sum(r['improved:'+k]['mean']['MAPE'] < r[k]['mean']['MAPE'] for k in FOUR)} of 4 models and raises mean R2 for {sum(r['improved:'+k]['mean']['R2'] > r[k]['mean']['R2'] for k in FOUR)} of 4 (largest gain: {max(FOUR, key=lambda k: r['improved:'+k]['mean']['R2'] - r[k]['mean']['R2'])}); where it does not help (XGBoost: {r['XGBoost']['mean']['R2']:.2f} vs {r['improved:XGBoost']['mean']['R2']:.2f}) the difference is within split noise.
- RMSE is larger than the paper's ({r['XGBoost']['mean']['RMSE']:.0f} vs {P['paper']['XGBoost']['RMSE']:.0f} cycles for XGBoost) partly because our cells span up to 1,935 cycles.
- **Not reproducible with the data we have:** the paper's 108 ICA features and its claim that mid-charge-voltage features dominate importance. That requires the raw MIT-Stanford voltage-capacity curves (see `tools/convert_severson.py` for the route once they are available).

## 2. How far the MIT-Stanford accuracy generalises

| Validation | {best} R2 |
|---|---|
| Random-split CV (5-fold x 3) | {M['cv']['mit_all'][best]['R2']:.2f} |
| Leave-protocol-out (whole charging protocols held out) | {M['protocol_cv'][best]['R2']:.2f} |
| Leave-batch-out (mean of 3 batches) | {np.mean([M['cross_batch'][h]['models'][best]['R2'] for h in M['cross_batch']]):.2f} |

![generalization](figures/04_mit_generalization.png)

Accuracy falls as validation gets stricter; on a whole unseen batch every model scores below zero. Nominal 80% intervals cover {M['interval']['empirical_coverage']*100:.0f}% of cells. Tier (GOOD / NEEDS TESTING / POOR) agreement is {M['tier_accuracy']*100:.0f}%.

## 3. The paper's method on real ICA: NASA PCoE cells (state of health)

Real dQ/dV from the constant-current charge, plus charge/constant-voltage and temperature features, each also as a change from the cell's own first 5 cycles. {N['meta']['cells_ok']} of {N['meta']['cells_total']} cells pass a quality gate. Leave-battery-out CV:

| Model | R2 | RMSE (SOH points) |
|---|---|---|
""" + "\n".join(f"| {k} | {v['R2']:.2f} ± {v['R2_std']:.2f} | {v['RMSE']:.2f} |" for k, v in N["soh"]["cv"].items()) + f"""

![nasa scatter](figures/06_nasa_soh_scatter.png)
![nasa ablation](figures/07_nasa_ablation.png)
![nasa ica](figures/08_nasa_ica_B0005.png)

- Most of the signal comes from changes in charge time and constant-voltage-phase behaviour; absolute ICA features alone are weak (R2 {N['soh']['ablation'][nb]['absolute ICA only (+ambient)']['R2']:.2f}).
- Remaining life on this data does **not** beat a cycle-count baseline, and the paper's first-50-cycles life prediction is not trainable here (only {N['rul']['n_cells']} usable cells reach 80% SOH, {N['rul']['cells_ending_before_50']} of them before cycle 50).

## 4. Accelerated-life battery packs ({A['meta']['packs']} packs, {A['meta']['cycles']:,} cycles)

| Task | Best model | R2 | Error | Baseline (cycle count + load) |
|---|---|---|---|---|
| SOH from charge-side signals | {ab_} | {ac['R2']:.2f} | {ac['RMSE']:.1f} SOH points | R2 {A['soh']['ablation'][ab_]['cycle number + load only']['R2']:.2f} |
| Remaining life to 80% capacity ({A['rul']['n_packs']} packs) | {rb} | {rc['R2']:.2f} | {rc['RMSE']:.0f} cycles | R2 {base['R2']:.2f} |
| Second-life transfer (3 packs) | {ab_} | {tr.get('R2', float('nan')):.2f} | {tr.get('RMSE', float('nan')):.1f} SOH points | n/a |

![alt ica](figures/09_alt_ica_aging_pack01.png)
![alt soh](figures/10_alt_soh_scatter.png)
![alt rul](figures/11_alt_rul_scatter.png)
![life vs load](figures/12_alt_life_vs_load.png)

Life falls roughly as load^{lv['slope']:.2f} (only {lv['n']} packs reached 80%; leave-one-out error {lv['loo_mape']:.0f}%). Remaining life beats its baseline but underestimates the longest-lived packs. Second-life transfer is weak.

## 5. Reference cell

![sp20](figures/13_sp20_reference.png)

The SP20-1 workbook is one reference test: initial capacity {SP['initial_capacity_ah']:.2f} Ah. It is shown as reference data, not used for training.

## What is and is not claimed
- Replicated: the paper's evaluation protocol and four-model comparison, on our data.
- Not replicated: the paper's exact numbers, which need its 124 cells and raw-curve ICA features.
- All other claims above are supported by cross-validated results in `backend/artifacts/` and visible in the web app.
"""
(OUT / "RESULTS.md").write_text(md)
print("wrote", len(list(FIG.glob('*.png'))), "figures,", len(list(TAB.glob('*.csv'))), "tables, RESULTS.md")
