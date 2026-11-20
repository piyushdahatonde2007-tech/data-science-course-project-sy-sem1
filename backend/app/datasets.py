"""Dataset registry: built-in MIT-Stanford batches + user-uploaded CSVs."""
import re
import numpy as np
import pandas as pd
from . import config as C
from .features import CORE, feature_table

REQUIRED = ["battery_id", "cycle"] + CORE
BUILTIN = {
    "mit_all": ("MIT-Stanford — all batches", None,
                "All 3 experimental batches combined (LFP/graphite 18650, fast-charge protocols)."),
    "mit_b1": ("MIT-Stanford — batch 1", "1", "Batch 1 cells (2017-05)."),
    "mit_b2": ("MIT-Stanford — batch 2", "2", "Batch 2 cells (2017-06). Shorter lives overall."),
    "mit_b3": ("MIT-Stanford — batch 3", "3", "Batch 3 cells (2018-04). Longest-lived cells."),
}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "upload"


class Registry:
    def __init__(self):
        self.raw = pd.read_csv(C.RAW_CSV)
        self.frames = {}   # dataset id -> raw dataframe
        self.feat = {}     # dataset id -> feature table
        self.meta = {}     # dataset id -> info
        full = feature_table(self.raw)
        self.full = full
        for did, (name, batch, desc) in BUILTIN.items():
            sub = full if batch is None else full[full.batch == batch]
            self.frames[did] = self.raw[self.raw.battery_id.isin(sub.battery_id)]
            self.feat[did] = sub.reset_index(drop=True)
            self.meta[did] = dict(id=did, name=name, description=desc, builtin=True, source="MIT-Stanford (from project zip)")
        for p in sorted(C.UPLOAD_DIR.glob("*.csv")):
            try:
                self._add(p.stem, pd.read_csv(p), name=p.stem, persist=False)
            except Exception as e:  # skip broken uploads, keep server alive
                print(f"[warn] skipping upload {p.name}: {e}")
        for did in self.meta:
            self._stats(did)

    def _stats(self, did):
        t = self.feat[did]
        lab = t.cycle_life.dropna()
        m = self.meta[did]
        m.update(n_cells=int(len(t)), labeled=bool(len(lab) == len(t) and len(t) > 0),
                 life_min=float(lab.min()) if len(lab) else None,
                 life_median=float(lab.median()) if len(lab) else None,
                 life_max=float(lab.max()) if len(lab) else None)

    def _add(self, did, df, name, persist=True):
        miss = [c for c in REQUIRED if c not in df.columns]
        if miss:
            raise ValueError(f"missing columns: {miss}. Required: {REQUIRED}; optional: C1, Q1, C2, cycle_life")
        df = df.copy()
        df["battery_id"] = df["battery_id"].astype(str)
        for c in df.columns:
            if c != "battery_id":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        t = feature_table(df)
        if t.empty:
            raise ValueError("no usable cells (need >=10 valid cycles per battery_id, with QC>0 and QD>0)")
        self.frames[did], self.feat[did] = df[df.battery_id.isin(t.battery_id)], t
        self.meta[did] = dict(id=did, name=name, description="User-uploaded dataset.", builtin=False,
                              source="upload")
        if persist:
            df.to_csv(C.UPLOAD_DIR / f"{did}.csv", index=False)
        self._stats(did)
        return did

    def add_upload(self, filename, df):
        base = slug(filename.rsplit(".", 1)[0])
        did, i = f"up_{base}", 1
        while did in self.meta and self.meta[did]["builtin"] is False and False:
            i += 1
        return self._add(did, df, name=filename)

    def delete_upload(self, did):
        if did not in self.meta or self.meta[did]["builtin"]:
            raise KeyError(did)
        for d in (self.frames, self.feat, self.meta):
            d.pop(did, None)
        (C.UPLOAD_DIR / f"{did}.csv").unlink(missing_ok=True)

    def cell_curves(self, did, bid):
        g = self.frames[did]
        g = g[g.battery_id == bid].sort_values("cycle")
        if g.empty:
            return None
        cols = ["cycle", "QD", "QC", "IR", "Tavg", "chargetime"]
        return {c.lower(): [None if pd.isna(v) else float(v) for v in g[c]] for c in cols}
