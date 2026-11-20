"""NASA/UCF accelerated life-testing packs (Fricke, Nascimento, Viana 2023): per-discharge-cycle table.

Each CSV logs one battery pack continuously from start to failure. Modes: -1 discharge, 0 rest, 1 charge. Discharge missions are
either 'reference' (constant 2.5 A, mission_type 0) or 'regular' (mission_type 1, load per pack). Charging has only a voltage
trace (no current), so charge-side features are time/voltage/temperature based."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew
from . import config as C

# ICA of the constant-2.5 A reference discharge (pack voltage, 2 cells in series): |dQ/dV| on a fixed grid
V_LO, V_HI = 5.80, 8.10
VGRID = np.round(np.arange(V_LO, V_HI + 1e-9, 0.01), 3)
INTERIOR = (VGRID >= 5.9) & (VGRID <= 8.0)      # drop 0.1 V at each end: the smoothing filter distorts the edges
VPLOT = VGRID[INTERIOR]
IC_POINTS = [6.0, 6.3, 6.6, 6.9, 7.2, 7.5, 7.8, 8.0]
ICA_COLS = [f"ic_v{int(round(v * 100))}" for v in IC_POINTS] + ["ica_max", "ica_argmax_v", "ica_mean", "ica_std", "ica_skew", "ica_kurt", "ica_slope", "ica_dq_window"]


def ica_ref(tt, ii, vv):
    """Reference-discharge ICA: returns (smoothed |dQ/dV| on VGRID, feature dict) or None if the voltage window is not covered."""
    ok = np.isfinite(ii) & np.isfinite(vv) & (vv > 1.0)
    if ok.sum() < 50:
        return None
    t, I, V = tt[ok], np.abs(ii[ok]), vv[ok]
    Q = np.r_[0.0, np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))] / 3600.0
    Vf, Qf = np.minimum.accumulate(V)[::-1], Q[::-1]
    Vu, ix = np.unique(Vf, return_index=True)
    if len(Vu) < 20 or Vu[0] > V_LO or Vu[-1] < V_HI:
        return None
    q = np.interp(VGRID, Vu, Qf[ix])
    ic = savgol_filter(np.abs(np.gradient(q, VGRID)), 21, 3)[INTERIOR]
    qi = q[INTERIOR]
    i = int(np.argmax(ic))
    f = {f"ic_v{int(round(v * 100))}": float(ic[int(np.argmin(np.abs(VPLOT - v)))]) for v in IC_POINTS}
    f.update(ica_max=float(ic[i]), ica_argmax_v=float(VPLOT[i]), ica_mean=float(ic.mean()), ica_std=float(ic.std()), ica_skew=float(skew(ic)),
             ica_kurt=float(kurtosis(ic)), ica_slope=float(np.polyfit(VPLOT, ic, 1)[0]), ica_dq_window=float(qi[-1] - qi[0]))
    return ic, f

COLS = ["time", "mode", "voltage_charger", "temperature_battery", "voltage_load", "current_load", "mission_type"]

# Loads from the dataset READMEs (A). Constant packs keep one level; recommissioned/second-life switch levels.
LOADS = {
    "01": ("regular", "constant", [9.3]), "11": ("regular", "constant", [9.3]), "31": ("regular", "constant", [12.9]),
    "22": ("regular", "constant", [12.9]), "23": ("regular", "constant", [14.3]), "52": ("regular", "constant", [14.3]),
    "00": ("regular", "constant", [16.0]), "10": ("regular", "constant", [16.0]), "20": ("regular", "constant", [19.0]),
    "30": ("regular", "constant", [19.0]), "21": ("regular", "constant", [19.0]),
    "41": ("regular", "variable", [14.3]), "51": ("regular", "variable", [14.3]),
    "40": ("regular", "variable", [17.0]), "50": ("regular", "variable", [17.0]),
    "32": ("recommissioned", "switch", [16.0, 14.3]), "53": ("recommissioned", "switch", [14.3, 16.0]),
    "02": ("recommissioned", "switch", [16.0, 12.9]), "33": ("recommissioned", "switch", [16.0, 12.9]),
    "12": ("recommissioned", "switch", [16.0, 9.3]), "24": ("recommissioned", "switch", [16.0, 9.3]),
    "03": ("recommissioned", "switch", [16.0, 14.3, 7.5]), "25": ("recommissioned", "switch", [16.0, 7.5, 14.3]),
    "13": ("second_life", "switch", [16.0, 9.3]), "36": ("second_life", "switch", [14.3, 7.5]), "54": ("second_life", "switch", [12.9, 5.0]),
}


def _first_valid(x):
    x = x[np.isfinite(x) & (x > 1.0)]
    return float(x[0]) if len(x) else np.nan


def extract_pack(path):
    """One pack CSV -> (DataFrame, ICA curves). One row per discharge episode with the preceding charge/rest summaries;
    reference discharges (mission 0) also get ICA features and a stored dQ/dV curve."""
    d = pd.read_csv(path, usecols=COLS)
    for c in COLS:                      # a few files contain stray text in numeric columns
        if d[c].dtype == object:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    t, m = d["time"].to_numpy(), d["mode"].to_numpy()
    V, I, T = d["voltage_load"].to_numpy(), d["current_load"].to_numpy(), d["temperature_battery"].to_numpy()
    vc, mt = d["voltage_charger"].to_numpy(), d["mission_type"].to_numpy()
    cut = np.r_[0, np.flatnonzero(np.diff(m) != 0) + 1, len(m)]
    rows, curves, last_ch, last_rest, n = [], [], None, None, 0
    for a, b in zip(cut[:-1], cut[1:]):
        mode = m[a]
        if mode == 1 and b - a >= 20:
            tc, v = t[a:b], vc[a:b]
            ok = np.isfinite(v) & (v > 1.0)
            if ok.sum() >= 10:
                q = np.percentile(tc[ok] - tc[ok][0], [25, 50, 75])
                vv = v[ok]
                last_ch = dict(ch_dur_s=float(tc[-1] - tc[0]), ch_v_start=float(vv[0]), ch_v_end=float(vv[-1]), ch_v_max=float(vv.max()),
                               ch_v_q25=float(np.interp(q[0], tc[ok] - tc[ok][0], vv)), ch_v_q50=float(np.interp(q[1], tc[ok] - tc[ok][0], vv)),
                               ch_v_q75=float(np.interp(q[2], tc[ok] - tc[ok][0], vv)), ch_t_mean=float(np.nanmean(T[a:b])), ch_t_max=float(np.nanmax(T[a:b])))
        elif mode == 0 and b - a >= 5:
            r = _first_valid(vc[a:b])
            if np.isfinite(r):
                last_rest = dict(rest_v=r, rest_s=float(t[b - 1] - t[a]))
        elif mode == -1 and b - a >= 20:
            tt, ii, vv, tp = t[a:b], I[a:b], V[a:b], T[a:b]
            good = np.isfinite(ii) & np.isfinite(tt)
            if good.sum() < 20:
                continue
            n += 1
            vg = vv[np.isfinite(vv) & (vv > 1.0)]
            mission = int(np.nanmedian(mt[a:b])) if np.isfinite(mt[a:b]).any() else -1
            ica = ica_ref(tt, ii, vv) if mission == 0 else None
            if ica is not None:
                curves.append(dict(k=n, ic=[round(float(x), 3) for x in ica[0]]))
            rows.append(dict(**(ica[1] if ica is not None else {}), cycle=n, t_start=float(tt[0]), mission=mission,
                             dur_s=float(tt[-1] - tt[0]), cap_ah=float(np.trapezoid(np.abs(np.nan_to_num(ii)), tt) / 3600),
                             i_mean=float(np.nanmean(np.abs(ii))), i_std=float(np.nanstd(ii)), v_start=float(vg[0]) if len(vg) else np.nan,
                             v_min=float(vg.min()) if len(vg) else np.nan, t_mean=float(np.nanmean(tp)), t_max=float(np.nanmax(tp)),
                             **(last_ch or {}), **(last_rest or {})))
    return pd.DataFrame(rows), curves
