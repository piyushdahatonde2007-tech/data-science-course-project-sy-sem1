"""Early-cycle degradation features from per-cycle summary data (window-aware)."""
import numpy as np
import pandas as pd

CORE = ["IR", "QC", "QD", "Tavg", "Tmin", "Tmax", "chargetime"]
PROTOCOL_RAW = ["C1", "Q1", "C2"]
PROTOCOL_COLS = ["c1_rate", "q1_switch_pct", "c2_rate"]

CATEGORIES = {
    "Capacity fade": ["qd_mean", "qd_start", "qd_end", "qd_slope", "qd_delta", "qd_std", "qd_min",
                      "qc_mean", "qc_slope", "qc_std", "coulombic_eff_mean"],
    "Internal resistance": ["ir_mean", "ir_start", "ir_end", "ir_slope", "ir_delta", "ir_std", "ir_min", "ir_max"],
    "Statistical moments": ["qd_variance", "qd_skewness", "qd_kurtosis"],
    "Inter-cycle variance": ["log_intercycle_var_qd", "intercycle_var_ir"],
    "Mid-window": ["mid_qd_slope", "mid_ir_slope", "mid_qd_mean", "mid_ir_mean"],
    "Thermal": ["tavg_mean", "tavg_slope", "tmax_mean", "tmax_max", "trange_mean", "temp_ir_corr"],
    "Charge time": ["chargetime_mean", "chargetime_slope", "chargetime_std"],
    "Protocol": PROTOCOL_COLS,
}
DEG_COLS = [c for k, v in CATEGORIES.items() if k != "Protocol" for c in v]
FEATURE_COLS = DEG_COLS + PROTOCOL_COLS
CAT_OF = {c: k for k, v in CATEGORIES.items() for c in v}


def clean_cell(g: pd.DataFrame, window=None, min_rows=10):
    """Return dict of float arrays for one cell, or None if unusable."""
    g = g.sort_values("cycle")
    if window:
        g = g[g["cycle"] <= window]
    g = g.dropna(subset=["cycle"] + CORE)
    g = g[(g["QC"] > 0) & (g["QD"] > 0)]
    if len(g) < min_rows:
        return None
    return {k: g[k].to_numpy(float) for k in ["cycle"] + CORE}


def protocol_of(g: pd.DataFrame):
    out = []
    for c in PROTOCOL_RAW:
        v = g[c].dropna() if c in g else []
        out.append(float(v.iloc[0]) if len(v) else np.nan)
    return tuple(out)


def extract_features(s: dict, protocol=(np.nan, np.nan, np.nan)) -> dict:
    cyc, ir, qc, qd = s["cycle"], s["IR"], s["QC"], s["QD"]
    tavg, tmin, tmax, ct = s["Tavg"], s["Tmin"], s["Tmax"], s["chargetime"]
    n = len(cyc)

    def slope(y, x=cyc):
        return float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 and np.ptp(x) > 0 else 0.0

    def delta(y):
        m = max(1, n // 5)
        return float(np.mean(y[-m:]) - np.mean(y[:m]))

    f = {}
    f.update(qd_mean=np.mean(qd), qd_start=qd[0], qd_end=qd[-1], qd_slope=slope(qd), qd_delta=delta(qd),
             qd_std=np.std(qd), qd_min=np.min(qd), qc_mean=np.mean(qc), qc_slope=slope(qc), qc_std=np.std(qc),
             coulombic_eff_mean=np.mean(qd / qc))
    f.update(ir_mean=np.mean(ir), ir_start=ir[0], ir_end=ir[-1], ir_slope=slope(ir), ir_delta=delta(ir),
             ir_std=np.std(ir), ir_min=np.min(ir), ir_max=np.max(ir))
    r = qd - np.polyval(np.polyfit(cyc, qd, 1), cyc)
    v = np.var(r) + 1e-12
    f.update(qd_variance=np.var(r), qd_skewness=np.mean(r ** 3) / v ** 1.5, qd_kurtosis=np.mean(r ** 4) / v ** 2 - 3)
    k = max(1, int(round(0.18 * n)))  # lag ~9 cycles for a 50-cycle window
    f["log_intercycle_var_qd"] = float(np.log10(np.var(qd[k:] - qd[:-k]) + 1e-12)) if n > k else 0.0
    f["intercycle_var_ir"] = float(np.var(np.diff(ir))) if n > 1 else 0.0
    span = np.ptp(cyc)
    m = (cyc >= cyc[0] + 0.3 * span) & (cyc <= cyc[0] + 0.7 * span)
    if m.sum() > 2:
        f.update(mid_qd_slope=slope(qd[m], cyc[m]), mid_ir_slope=slope(ir[m], cyc[m]),
                 mid_qd_mean=np.mean(qd[m]), mid_ir_mean=np.mean(ir[m]))
    else:
        f.update(mid_qd_slope=0.0, mid_ir_slope=0.0, mid_qd_mean=f["qd_mean"], mid_ir_mean=f["ir_mean"])
    f.update(tavg_mean=np.mean(tavg), tavg_slope=slope(tavg), tmax_mean=np.mean(tmax), tmax_max=np.max(tmax),
             trange_mean=np.mean(tmax - tmin),
             temp_ir_corr=float(np.corrcoef(tavg, ir)[0, 1]) if np.std(tavg) > 0 and np.std(ir) > 0 else 0.0)
    f.update(chargetime_mean=np.mean(ct), chargetime_slope=slope(ct), chargetime_std=np.std(ct))
    f.update(zip(PROTOCOL_COLS, protocol))
    return {c: float(f[c]) for c in FEATURE_COLS}


def feature_table(df: pd.DataFrame, window=None, min_rows=10) -> pd.DataFrame:
    """One row per cell: battery_id, batch, cycle_life (NaN if unlabeled), FEATURE_COLS."""
    rows = []
    for bid, g in df.groupby("battery_id", sort=True):
        s = clean_cell(g, window, min_rows)
        if s is None:
            continue
        row = {"battery_id": str(bid), "n_cycles": len(s["cycle"]), "max_cycle": float(s["cycle"].max())}
        life = g["cycle_life"].dropna() if "cycle_life" in g else []
        row["cycle_life"] = float(life.iloc[0]) if len(life) else np.nan
        row.update(extract_features(s, protocol_of(g)))
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["batch"] = out["battery_id"].str.extract(r"^b(\d+)c")[0].fillna("custom")
    return out
