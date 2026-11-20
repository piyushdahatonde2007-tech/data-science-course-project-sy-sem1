#!/usr/bin/env python3
"""Write data_summary/: one summarised table per dataset plus an inventory of what raw data was dropped and why.
Usage (from the project root): python tools/make_data_summary.py"""
import json, shutil, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.datasets import Registry  # noqa: E402

ART, OUT = ROOT / "backend" / "artifacts", ROOT / "data_summary"
OUT.mkdir(exist_ok=True)
N, A = json.loads((ART / "nasa_results.json").read_text()), json.loads((ART / "alt_results.json").read_text())
reg = Registry()
reg.feat["mit_all"].drop(columns=["max_cycle"], errors="ignore").to_csv(OUT / "mit_stanford_per_cell.csv", index=False)
shutil.copy(ART / "nasa_features.csv", OUT / "nasa_pcoe_per_cycle.csv")
pd.DataFrame(N["cells"]).to_csv(OUT / "nasa_pcoe_cells_quality_gate.csv", index=False)
shutil.copy(ART / "alt_cycles.csv", OUT / "alt_packs_per_cycle.csv")
pd.DataFrame([{**p, "loads": "/".join(map(str, p["loads"]))} for p in A["packs"]]).to_csv(OUT / "alt_packs_summary.csv", index=False)
(OUT / "sp20_1_reference_test.json").write_text(json.dumps(N["sp1"]))
rows = [
    dict(dataset="MIT-Stanford (Severson et al. 2019), summary level", cells="138 (46/48/44 in batches 1/2/3)", original="early_50_cycles.csv, 6,854 rows (from the original project zip)",
         kept_in_package="backend/data/raw/mit_stanford_early50.csv and data_summary/mit_stanford_per_cell.csv (40 features per cell)", dropped="nothing", why="Raw within-cycle voltage/capacity curves were never supplied, so true ICA on this dataset is not possible yet."),
    dict(dataset="NASA PCoE battery aging (cleaned version)", cells="34 (21 pass the quality gate)", original="7,565 cycle CSVs (2,815 charge, 2,794 discharge, 1,956 impedance), 585 MB unzipped, 228 MB zipped",
         kept_in_package="metadata.csv, group READMEs, data_summary/nasa_pcoe_per_cycle.csv (2,768 charge cycles: capacity, SOH label, ICA/charge/temperature features), 229 cached dQ/dV curves",
         dropped="the 7,565 raw cycle CSVs", why="The per-cycle table holds every value the models use; discharge and impedance files were never used. Originals stay with the data owner."),
    dict(dataset="SP20-1 initial-capacity workbook", cells="1 cell", original="one Arbin Excel export, 27,602 rows", kept_in_package="the workbook itself and data_summary/sp20_1_reference_test.json (steps, profile, ICA, 2.009 Ah)", dropped="nothing", why="Small enough to keep whole."),
    dict(dataset="NASA/UCF accelerated-life packs (Fricke, Nascimento, Viana 2023; MIT licence)", cells="26 packs (15 regular, 8 recommissioned, 3 second-life)", original="26 pack CSVs, about 3.4 GB unzipped, 533 MB zipped",
         kept_in_package="data_summary/alt_packs_per_cycle.csv (9,614 discharge cycles with charge/rest/temperature features and ICA features of 448 reference discharges), 448 cached dQ/dV curves, pack summary",
         dropped="the 26 raw pack CSVs", why="Re-extractable with tools/extract_alt.py if the raw files are restored; all model inputs are in the per-cycle table."),
]
inv = pd.DataFrame(rows); inv.to_csv(OUT / "dataset_inventory.csv", index=False)
md = ["# Data inventory", "", "Summarised data for every dataset used. Raw files that the models do not need at run time were removed to keep the package small; each row says exactly what was kept and what was dropped.", ""]
for r in rows:
    md += [f"## {r['dataset']}", f"- Cells/packs: {r['cells']}", f"- Original: {r['original']}", f"- Kept here: {r['kept_in_package']}", f"- Dropped: {r['dropped']}", f"- Why: {r['why']}", ""]
md += ["## Files in this folder", *[f"- `{p.name}` ({p.stat().st_size / 1e6:.1f} MB)" for p in sorted(OUT.glob('*')) if p.name != 'DATA_INVENTORY.md']]
(OUT / "DATA_INVENTORY.md").write_text("\n".join(md) + "\n")
print("data_summary:", [p.name for p in sorted(OUT.glob('*'))])
