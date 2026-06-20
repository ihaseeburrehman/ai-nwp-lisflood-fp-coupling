# AI weather models → LISFLOOD-FP flood coupling (July 2021 Alzette flood)

Code and processed data accompanying:

> *Coupling data-driven NWP (GraphCast, FuXi, AIFS) to high-resolution hydraulic
> flood simulation: the July 2021 Luxembourg flood.* (Journal of Hydrology, under review)

This repository contains the **coupling and analysis pipeline** — the part of the study
that is *not* already public. The weather models, the hydraulic engine, and the input
datasets are all openly available from their original sources (see
[Inputs not included](#inputs-not-included)); what is released here is the glue that
turns each model's precipitation forecast into a flood simulation and the analysis that
produces every figure in the paper.

```
 NWP precip forecast            forcing/                LISFLOOD-FP (FV1+CUDA, 10 m)
 (GraphCast / FuXi /  ──▶  *_to_lisflood_rain.py  ──▶   config/  (one run per model)  ──┐
  AIFS / WRF d01,d03)        → rain_<model>.nc          sim_time = 96 h window           │
                                                                                         ▼
        figures/tables  ◀──  analysis/plot_and_stats.py  ◀──  analysis/extract_station_discharge.py
        (NSE/KGE/PDE/PTE,        (metrics + hydrographs           (per-gauge discharge & depth
         Taylor, hydrographs)     + Taylor, over 96 h)             from 6-hourly Qx/Qy/wd grids)
```

## Repository layout

| Path | Contents |
|---|---|
| `forcing/ai_to_lisflood_rain.py`  | Convert GraphCast/FuXi/AIFS 6-hourly QPF to a LISFLOOD-FP `dynamicrainfile` on the 10 m grid |
| `forcing/wrf_to_lisflood_rain.py` | Same, for WRF `wrfout` (d01 = 12 km, d03 = 1.3 km) total precipitation |
| `config/lisflood/*.par`           | LISFLOOD-FP parameter files (one per forcing) |
| `config/slurm/run_*.sh`           | SLURM batch scripts (1× NVIDIA A100 each) |
| `analysis/extract_station_discharge.py` | Per-gauge discharge/depth from the 6-hourly output grids |
| `analysis/plot_and_stats.py`      | Skill/error metrics, hydrographs, water level, Taylor diagram |
| `data/processed/`                 | Per-gauge discharge CSVs + merged/averaged CSVs — enough to **reproduce every figure without rerunning the models or LISFLOOD-FP** |
| `lisflood_patches/`               | The three FV1-CUDA large-grid patches against LISFLOOD-FP v8.2 (see its README) |

## Reproducibility — exact configuration

**Event / window.** Major flood of 13–17 July 2021, Alzette basin, Luxembourg
(468 km² drainage area at 10 m resolution; DEM in EPSG:2169). Hydraulic runs span
13 Jul 00:00 → 17 Jul 06:00 UTC; **all models are intercompared over the common 96 h
flood window 13 Jul 00:00 → 17 Jul 00:00 UTC**, 6-hourly.

**Forecast models** (rapid-update cycle; each `forecast_<YYYYMMDDTHH>.nc` holds the 6 h
precipitation accumulation valid at the filename timestamp):
- **GraphCast** — Google DeepMind, trained on ERA5 to ~2017 (2021 out-of-sample).
- **FuXi** — via the ECMWF `ai-models` framework; ERA5-trained to ~2017 (2021 out-of-sample).
- **AIFS Single 1.0** — ECMWF; pre-trained on ERA5 1979–2022 and fine-tuned on operational
  IFS analyses 2016–2022 (**2021 is in-sample → data-leakage caveat**).
- **WRF 4.5 + WRFDA "After-DA"** — convection-permitting d03 = 1.3 km and d01 = 12 km,
  ERA5 initial/boundary conditions, observation-corrected (see first companion study).

**Hydraulic model.** LISFLOOD-FP v8.2, FV1 + CUDA solver, single NVIDIA A100-40 GB,
10 m grid (2440 × 4362 ≈ 10.6 M cells). Key `.par` settings: `fv1`, `cuda`,
`sim_time 367200` (102 h), `saveint 21600` (6 h output), warm start from
`july_13_18hr.wd`.

**Gauges** (EPSG:2169, in `extract_station_discharge.py`): Walferdange (water level);
Steinsel, Pfaffenthal, Livange, Hesperange (discharge).

## How to run

```bash
# 0. environment
conda create -n ainwp python=3.11 && conda activate ainwp
pip install -r requirements.txt

# 1. build the forcing for each model (run once per model)
#    AI:  python forcing/ai_to_lisflood_rain.py  <model> <forecast_dir> <dem.asc> rain_<model>.nc
#    WRF: python forcing/wrf_to_lisflood_rain.py <wrf_dir> <d01|d03>    <dem.asc> rain_<model>.nc
#    (window is fixed in the scripts: START=2021-07-13 00:00, END=2021-07-17 06:00)

# 2. run LISFLOOD-FP on the HPC (1 GPU per model)
sbatch config/slurm/run_aifs.sh        # etc. for wrf, wrf12, graphcast, fuxi

# 3. extract per-gauge discharge from the output grids
python analysis/extract_station_discharge.py /path/to/AI_flood_2021

# 4. metrics + all figures (reads data/processed/, writes figures + averaged_metrics.csv)
python analysis/plot_and_stats.py
```

To **reproduce the figures only** (no HPC, no model runs), the merged CSVs in
`data/processed/pgfplots/` already embed the observed series and all model series, so the
manuscript figures regenerate directly from them. Note that `analysis/plot_and_stats.py`
*recomputes* metrics from the raw gauge observations and so has machine-specific input
paths at the top of the file (`OBS_FILES`, `COMPANION_CSV`); set these to local copies of
the observation files if you re-run the full metric computation.

> **SLURM note:** the batch scripts use `#!/bin/bash` (a *non-login* shell) deliberately.
> On the ECMWF Atos system a login shell (`-l`) redirects stdout and, with `set -e`, kills
> the job within seconds.

## Inputs not included (public sources)

Model code/weights and raw inputs are **not** redistributed here; they are large and
already public. Regenerate or download them from:

- LISFLOOD-FP v8.2 — https://zenodo.org/records/13121102 (apply the patches in `lisflood_patches/`)
- WRF / WRFDA 4.5 — https://github.com/wrf-model/WRF
- ERA5 — https://cds.climate.copernicus.eu
- GraphCast — https://github.com/google-deepmind/graphcast
- AIFS Single 1.0 — https://huggingface.co/ecmwf/aifs-single-1.0
- FuXi (ai-models) — https://github.com/ecmwf-lab/ai-models-fuxi

Gauge **discharge and water-level observations** are from the Luxembourg water authority
(Administration de la gestion de l'eau, AGE); they are embedded in the merged CSVs under
`data/processed/pgfplots/` for figure reproduction and are available from AGE on request.

The intermediate rain NetCDFs (`rain_*.nc`, ~20–120 MB each) and raw forecast files are
regenerable from the above with `forcing/` and are omitted to keep the archive light.

## Citation & license

Code: MIT (`LICENSE`). Processed data CSVs: CC-BY-4.0. If you use this pipeline, please
cite the paper above and this repository (DOI on the archived release).
