#!/usr/bin/env python3
"""Convert MIT-Stanford (Severson et al. 2019) batch pickles into this project's CSV schema.

Usage:
  python tools/convert_severson.py batch1.pkl batch2.pkl batch3.pkl --out merged.csv [--cycles 50] [--exclude b1c8,b1c10]

Input : the .pkl batch files produced by the dataset authors' LoadData script (dict: cell id -> {summary, cycle_life,
        charge_policy, cycles}). MATLAB .mat files must be converted to .pkl first with that script.
Output: CSV with battery_id, cycle, IR, QC, QD, Tavg, Tmin, Tmax, chargetime, C1, Q1, C2, cycle_life
        (first --cycles cycles per cell), ready for the Upload tab or for replacing backend/data/raw/.
Notes : apply the paper's cell exclusions / batch-continuation merges with the authors' LoadData script first, or pass
        --exclude. Only summary arrays are used here; within-cycle curves (needed for true ICA) are not converted.
"""
import argparse, pickle, re, sys
import numpy as np
import pandas as pd

CORE = ["IR", "QC", "QD", "Tavg", "Tmin", "Tmax", "chargetime"]
POLICY = re.compile(r"([\d.]+)C\((\d+)%\)-([\d.]+)C")


def parse_policy(p):
    m = POLICY.search(str(p)) if p is not None else None
    return tuple(float(x) for x in m.groups()) if m else (np.nan, np.nan, np.nan)


def scalar(v):
    return float(np.asarray(v, dtype=float).ravel()[0])


def convert(batches, cycles=50, exclude=()):
    frames, skipped, seen = [], [], set()
    for batch in batches:
        for cid, cell in batch.items():
            if cid in exclude:
                skipped.append((cid, "excluded"))
                continue
            if cid in seen:
                raise ValueError(f"duplicate cell id {cid} across batch files")
            seen.add(cid)
            try:
                summ, life = cell["summary"], scalar(cell["cycle_life"])
                arr = {k: np.asarray(summ[k], dtype=float).ravel() for k in CORE}
            except Exception as e:
                skipped.append((cid, f"missing field: {e}"))
                continue
            if not np.isfinite(life) or life <= 0:
                skipped.append((cid, "no valid cycle_life"))
                continue
            n = min(len(v) for v in arr.values())
            cyc = np.asarray(summ["cycle"], dtype=float).ravel()[:n] if "cycle" in summ else np.arange(1, n + 1, dtype=float)
            df = pd.DataFrame({"battery_id": cid, "cycle": cyc, **{k: v[:n] for k, v in arr.items()}})
            df = df[(df.cycle >= 1) & (df.cycle <= cycles)]
            if len(df) < 10:
                skipped.append((cid, "fewer than 10 usable cycles"))
                continue
            df["C1"], df["Q1"], df["C2"] = parse_policy(cell.get("charge_policy"))
            df["cycle_life"] = life
            frames.append(df)
    if not frames:
        raise ValueError("no cells converted; check the input files")
    return pd.concat(frames, ignore_index=True), skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batches", nargs="+", help="batch .pkl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=50)
    ap.add_argument("--exclude", default="", help="comma-separated cell ids to drop")
    a = ap.parse_args(argv)
    data = []
    for f in a.batches:
        with open(f, "rb") as fh:
            data.append(pickle.load(fh))
    df, skipped = convert(data, a.cycles, set(filter(None, a.exclude.split(","))))
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {df.battery_id.nunique()} cells, {len(df)} rows, cycles 1-{a.cycles}")
    for cid, why in skipped:
        print(f"  skipped {cid}: {why}")


if __name__ == "__main__":
    sys.exit(main())
