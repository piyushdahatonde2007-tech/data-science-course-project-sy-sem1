"""NASA PCoE Li-ion aging data: quality gate, SOH/EOL labels and real ICA (dQ/dV) features from charge curves.

Every cell was charged identically (1.5 A constant current to 4.2 V, then constant voltage to 20 mA), so charge-curve
features are comparable across cells even though discharge conditions (load, cut-off, temperature) differ. Capacity in Ah
is therefore NOT comparable between cells; state of health (SOH) is defined relative to each cell's own reference."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew
from . import config as C

V_LO, V_HI = 4.00, 4.19                    # ICA window: reached by ~94% of cycles (CC start voltage varies by cycle)
VGRID = np.round(np.arange(V_LO, V_HI + 1e-9, 0.002), 4)
IC_POINTS = [4.00, 4.03, 4.06, 4.09, 4.12, 4.15, 4.18]
EOL_FRACTION = 0.70                        # 30% fade, the dataset's own end-of-life definition
EOL80_FRACTION = 0.80                      # 20% fade, the definition used for the MIT-Stanford data and in the paper

ICA_COLS = [f"ic_v{int(round(v * 100))}" for v in IC_POINTS] + ["ica_max", "ica_argmax_v", "ica_mean", "ica_std",
                                                                 "ica_skew", "ica_kurt", "ica_slope", "ica_dq_window"]
CHARGE_COLS = ["cc_time_s", "cv_time_s", "cv_t_i05", "cv_t_i02", "cv_t_i005", "q_total_ah", "q_cc_ah", "q_cv_ah"]
TEMP_COLS = ["t_mean", "t_max", "t_rise", "ambient_c"]
FEATURE_COLS = ICA_COLS + CHARGE_COLS + TEMP_COLS

# Start-of-life-relative features: value minus the cell's own median over its first REF_CYCLES charge cycles (no labels used)
REF_CYCLES = 5
DELTA_BASE = ICA_COLS + CHARGE_COLS + ["t_mean", "t_max", "t_rise"]
DELTA_COLS = [c + "_d" for c in DELTA_BASE]
MODEL_COLS = FEATURE_COLS + DELTA_COLS


def add_delta(tbl):
    t = tbl.sort_values(["battery_id", "cycle_no"])
    base = t.groupby("battery_id").head(REF_CYCLES).groupby("battery_id")[DELTA_BASE].median()
    out = tbl.copy()
    out[DELTA_COLS] = pd.DataFrame(t[DELTA_BASE].to_numpy() - base.loc[t.battery_id].to_numpy(),
                                   index=t.index, columns=DELTA_COLS).loc[tbl.index]
    return out


def baseline_from_features(feature_dicts):
    """Median of each DELTA_BASE feature over a cell's start-of-life cycles (NaN-safe)."""
    df = pd.DataFrame(feature_dicts)[DELTA_BASE]
    return df.median(skipna=True)


def _trapz_cum(y, t):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])


def ica_curve(V, Q):
    """Q(V) on a fixed grid -> smoothed dQ/dV (Savitzky-Golay, order 3). Returns None if the CC run misses the window."""
    Vm = np.maximum.accumulate(V)
    Vu, idx = np.unique(Vm, return_index=True)
    if len(Vu) < 10 or Vu[0] > V_LO or Vu[-1] < V_HI:
        return None
    q = np.interp(VGRID, Vu, Q[idx])
    ic = np.gradient(q, VGRID)
    return savgol_filter(ic, 11, 3), float(q[-1] - q[0])


def charge_features(path, ambient):
    """One charge file -> dict of features (NaN where a feature is not available for that cycle)."""
    try:
        d = pd.read_csv(path, usecols=["Voltage_measured", "Current_measured", "Temperature_measured", "Time"])
    except Exception:
        return features_from_df(None, ambient)
    return features_from_df(d, ambient)


def features_from_df(d, ambient):
    f = {c: np.nan for c in FEATURE_COLS}
    f["ambient_c"] = float(ambient)
    if d is None:
        return f
    t, I, V, T = (d[c].to_numpy(float) for c in ["Time", "Current_measured", "Voltage_measured", "Temperature_measured"])
    if len(t) < 30 or not np.all(np.isfinite([t[-1], I.sum()])):
        return f
    Q = _trapz_cum(I, t) / 3600.0
    f["q_total_ah"] = float(Q[-1])
    f.update(t_mean=float(np.nanmean(T)), t_max=float(np.nanmax(T)), t_rise=float(np.nanmax(T) - T[0]))
    pos = I > 1.0
    if pos.sum() < 20:
        return f
    cc = I >= 0.9 * np.median(I[pos])
    dd = np.diff(np.r_[0, cc.astype(int), 0])
    s, e = np.where(dd == 1)[0], np.where(dd == -1)[0]
    k = int(np.argmax(e - s))
    a, b = s[k], e[k]
    f["cc_time_s"] = float(t[b - 1] - t[a])
    f["q_cc_ah"] = float(Q[b - 1] - Q[a])
    f["cv_time_s"] = float(t[-1] - t[b - 1])
    f["q_cv_ah"] = float(Q[-1] - Q[b - 1])
    for name, thr in (("cv_t_i05", 0.5), ("cv_t_i02", 0.2), ("cv_t_i005", 0.05)):
        j = np.where(I[b:] < thr)[0]
        if len(j):
            f[name] = float(t[b + j[0]] - t[b - 1])
    r = ica_curve(V[a:b], Q[a:b])
    if r is not None:
        ic, dq = r
        for v in IC_POINTS:
            f[f"ic_v{int(round(v * 100))}"] = float(ic[int(np.argmin(np.abs(VGRID - v)))])
        i = int(np.argmax(ic))
        f.update(ica_max=float(ic[i]), ica_argmax_v=float(VGRID[i]), ica_mean=float(ic.mean()), ica_std=float(ic.std()),
                 ica_skew=float(skew(ic)), ica_kurt=float(kurtosis(ic)), ica_slope=float(np.polyfit(VGRID, ic, 1)[0]),
                 ica_dq_window=dq)
    return f


def curve_for_plot(path):
    """dQ/dV vs V for one charge file (for the UI), or None."""
    d = pd.read_csv(path, usecols=["Voltage_measured", "Current_measured", "Time"])
    t, I, V = (d[c].to_numpy(float) for c in ["Time", "Current_measured", "Voltage_measured"])
    pos = I > 1.0
    if pos.sum() < 20:
        return None
    Q = _trapz_cum(I, t) / 3600.0
    cc = I >= 0.9 * np.median(I[pos])
    dd = np.diff(np.r_[0, cc.astype(int), 0])
    s, e = np.where(dd == 1)[0], np.where(dd == -1)[0]
    k = int(np.argmax(e - s))
    r = ica_curve(V[s[k]:e[k]], Q[s[k]:e[k]])
    return None if r is None else {"v": VGRID.tolist(), "ic": [float(x) for x in r[0]]}


# ---------- cell table, quality gate, labels ----------
def load_metadata():
    m = pd.read_csv(C.NASA_DIR / "metadata.csv")
    m["cap"] = pd.to_numeric(m["Capacity"], errors="coerce")
    return m.sort_values(["battery_id", "uid"]).reset_index(drop=True)


def clean_capacity(cap):
    """Drop impossible values and isolated spikes using a rolling median. Returns (kept mask, smoothed series)."""
    s = pd.Series(cap, dtype=float)
    valid = s.notna() & (s > 0.2) & (s < 3.0)
    rm = s.where(valid).rolling(5, center=True, min_periods=2).median()
    keep = valid & ((s - rm).abs() <= 0.15 * rm)
    sm = s.where(keep).rolling(5, center=True, min_periods=2).median()
    return keep.to_numpy(), sm.to_numpy()


def cell_quality(cap):
    """Quality gate for using a cell's capacity trajectory as an SOH label. Returns dict with ok/reason/ref/eol."""
    keep, sm = clean_capacity(cap)
    n_keep = int(keep.sum())
    out = dict(n_discharges=int(len(cap)), n_clean=n_keep, ref_capacity=None, ok=False, reason="", eol_cycle=None, eol80_cycle=None)
    if n_keep < 20:
        out["reason"] = "fewer than 20 usable discharge capacities"
        return out
    first = np.asarray(cap, float)[keep][:5]
    ref = float(np.median(first))
    out["ref_capacity"] = ref
    rel = np.asarray(cap, float)[keep] / ref
    if ref < 0.5:
        out["reason"] = f"reference capacity {ref:.2f} Ah is implausibly low"
    elif np.std(first / ref) > 0.06:
        out["reason"] = "unstable start (capacity still ramping in first cycles)"
    elif rel.max() > 1.15:
        out["reason"] = "capacity rises above 115% of reference (mixed or changing load)"
    elif n_keep / len(cap) < 0.7:
        out["reason"] = "more than 30% of capacity values unusable"
    else:
        out["ok"], out["reason"] = True, "ok"
        below = np.where(np.nan_to_num(sm / ref, nan=9) <= EOL_FRACTION)[0]
        out["eol_cycle"] = int(below[0]) + 1 if len(below) else None
        b80 = np.where(np.nan_to_num(sm / ref, nan=9) <= EOL80_FRACTION)[0]
        out["eol80_cycle"] = int(b80[0]) + 1 if len(b80) else None
    return out


def pair_cycles(m):
    """Pair each charge file with the next discharge of the same battery. Returns DataFrame of paired cycles."""
    rows = []
    for bid, g in m.groupby("battery_id", sort=True):
        g = g.sort_values("uid")
        disch = g[g.type == "discharge"]
        cap = disch["cap"].to_numpy(float)
        keep, sm = clean_capacity(cap)
        q = cell_quality(cap)
        ordinal = {u: i + 1 for i, u in enumerate(disch.uid)}
        capmap = dict(zip(disch.uid, zip(cap, keep)))
        pending = None
        for _, r in g.iterrows():
            if r.type == "charge":
                pending = r
            elif r.type == "discharge" and pending is not None:
                c, kp = capmap[r.uid]
                rows.append(dict(battery_id=bid, charge_file=pending.filename, charge_uid=int(pending.uid),
                                 discharge_uid=int(r.uid), cycle_no=ordinal[r.uid], ambient_c=float(pending.ambient_temperature),
                                 capacity=float(c) if np.isfinite(c) else np.nan, cap_ok=bool(kp),
                                 soh=(float(c) / q["ref_capacity"] * 100 if kp and q["ref_capacity"] else np.nan),
                                 cell_ok=q["ok"]))
                pending = None
    return pd.DataFrame(rows)
