"""Re-extract the per-cycle table and ICA curves from the raw accelerated-life pack CSVs.
Usage (from backend/):  python ../tools/extract_alt.py [path/to/battery_alt_dataset]
Writes backend/artifacts/alt_cycles.csv and alt_ica_curves.json (about 30 s), then run: python -m app.alt_train"""
import sys, time, glob, os
sys.path.insert(0, '.')
import pandas as pd
from joblib import Parallel, delayed
from app import alt
root = sys.argv[1] if len(sys.argv) > 1 else 'data/alt/battery_alt_dataset'
files = sorted(glob.glob(root + '/*/battery*.csv'), key=os.path.getsize, reverse=True)
def one(f):
    t = time.time(); pid = os.path.basename(f)[7:9]
    df, cur = alt.extract_pack(f); df.insert(0, 'pack', pid); df.insert(1, 'folder', os.path.basename(os.path.dirname(f)))
    print(pid, len(df), len(cur), 'ref curves', round(time.time() - t), 's', flush=True); return df, cur
t0 = time.time()
parts = Parallel(n_jobs=3)(delayed(one)(f) for f in files)
import json
out = pd.concat([p[0] for p in parts], ignore_index=True)
out.to_csv('artifacts/alt_cycles.csv', index=False)
json.dump({p[0].pack.iloc[0]: p[1] for p in parts if len(p[0])}, open('artifacts/alt_ica_curves.json', 'w'))
print('DONE', out.shape, round(time.time() - t0), 's', flush=True)
