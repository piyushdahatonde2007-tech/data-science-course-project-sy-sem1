# Data inventory

Summarised data for every dataset used. Raw files that the models do not need at run time were removed to keep the package small; each row says exactly what was kept and what was dropped.

## MIT-Stanford (Severson et al. 2019), summary level
- Cells/packs: 138 (46/48/44 in batches 1/2/3)
- Original: early_50_cycles.csv, 6,854 rows (from the original project zip)
- Kept here: backend/data/raw/mit_stanford_early50.csv and data_summary/mit_stanford_per_cell.csv (40 features per cell)
- Dropped: nothing
- Why: Raw within-cycle voltage/capacity curves were never supplied, so true ICA on this dataset is not possible yet.

## NASA PCoE battery aging (cleaned version)
- Cells/packs: 34 (21 pass the quality gate)
- Original: 7,565 cycle CSVs (2,815 charge, 2,794 discharge, 1,956 impedance), 585 MB unzipped, 228 MB zipped
- Kept here: metadata.csv, group READMEs, data_summary/nasa_pcoe_per_cycle.csv (2,768 charge cycles: capacity, SOH label, ICA/charge/temperature features), 229 cached dQ/dV curves
- Dropped: the 7,565 raw cycle CSVs
- Why: The per-cycle table holds every value the models use; discharge and impedance files were never used. Originals stay with the data owner.

## SP20-1 initial-capacity workbook
- Cells/packs: 1 cell
- Original: one Arbin Excel export, 27,602 rows
- Kept here: the workbook itself and data_summary/sp20_1_reference_test.json (steps, profile, ICA, 2.009 Ah)
- Dropped: nothing
- Why: Small enough to keep whole.

## NASA/UCF accelerated-life packs (Fricke, Nascimento, Viana 2023; MIT licence)
- Cells/packs: 26 packs (15 regular, 8 recommissioned, 3 second-life)
- Original: 26 pack CSVs, about 3.4 GB unzipped, 533 MB zipped
- Kept here: data_summary/alt_packs_per_cycle.csv (9,614 discharge cycles with charge/rest/temperature features and ICA features of 448 reference discharges), 448 cached dQ/dV curves, pack summary
- Dropped: the 26 raw pack CSVs
- Why: Re-extractable with tools/extract_alt.py if the raw files are restored; all model inputs are in the per-cycle table.

## Files in this folder
- `alt_packs_per_cycle.csv` (2.9 MB)
- `alt_packs_summary.csv` (0.0 MB)
- `dataset_inventory.csv` (0.0 MB)
- `mit_stanford_per_cell.csv` (0.1 MB)
- `nasa_pcoe_cells_quality_gate.csv` (0.0 MB)
- `nasa_pcoe_per_cycle.csv` (1.3 MB)
- `sp20_1_reference_test.json` (0.0 MB)
