#!/usr/bin/env python3
"""EFAS benchmark discharge at each gauge over the full 96 h flood window.

The EFAS columns of the merged CSVs previously stopped at 16 July 00:00, four
six-hourly steps short of the 96 h window, because the companion study
(rehman2024wrflisflood) plotted only that far. The source NetCDFs cover
10-19 July, so nothing needs re-downloading; this script re-extracts all 17
steps of 13-17 July.

Extraction logic is taken unchanged from the author's own
    Python scripts/lisflood_scripts/generate_lisflood_hydrographs.py
(function `extract_efas_data`): nearest grid point by Euclidean distance in
lat/lon after transforming the gauge from EPSG:2169, and the value at the
nearest `valid_time` of the (time, step) forecast array. Reproducing the 13
values already published is the built-in check (see --verify).

Reads:
  <EFAS_DIR>/driven_by_ECMWF.nc, driven_by_DWD.nc   (dis06, m3/s)
Writes:
  data/lisflood_96h/efas/efas_station_Q.csv

    python3 analysis/scripts/extract_efas_discharge.py [--verify]
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lisflood_96h" / "efas"

EFAS_DIR = Path(
    "/Volumes/SanDisk 2TB/2026-07-14-115148.previous/Data/Users/haseeb.rehman/"
    "Documents/Misc/Data_Datasets/Stations_and_Observations/"
    "EFAS_River_Discharge_forecasted_202107"
)
SOURCES = {"EFAS_ECMWF": "driven_by_ECMWF.nc", "EFAS_DWD": "driven_by_DWD.nc"}

# Gauge coordinates (EPSG:2169), identical to the LISFLOOD-FP extractor.
STATIONS = {
    "Walferdange": (77256, 81571),
    "Steinsel":    (77432, 82659),
    "Pfaffenthal": (77409, 76226),
    "Livange":     (76151, 65753),
    "Hesperange":  (78623, 72404),
}

# Option B window: 13 July 18:00 -> 17 July 18:00 UTC (96 h), matching the
# hydraulic runs, whose start grid is the model state at 13 July 18:00.
START = datetime(2021, 7, 13, 18, 0)
NSTEP, INTERVAL_H = 17, 6
TARGET = [START + timedelta(hours=INTERVAL_H * i) for i in range(NSTEP)]


def extract(path):
    """Nearest-cell, nearest-valid-time dis06 at every station, in m3/s."""
    ds = xr.open_dataset(path)
    tr = Transformer.from_crs("EPSG:2169", "EPSG:4326", always_xy=True)
    lats, lons = ds.latitude.values, ds.longitude.values
    vt = ds.valid_time.values
    vt_flat = vt.flatten()

    out = {}
    for name, (x, y) in STATIONS.items():
        lon_t, lat_t = tr.transform(x, y)
        dist = np.sqrt((lats - lat_t) ** 2 + (lons - lon_t) ** 2)
        yi, xi = np.unravel_index(np.argmin(dist), dist.shape)

        vals = []
        for t in TARGET:
            k = int(np.argmin(np.abs(vt_flat - np.datetime64(t))))
            ti, si = np.unravel_index(k, vt.shape)
            gap = abs(pd.Timestamp(vt[ti, si]) - pd.Timestamp(t))
            if gap > pd.Timedelta(0):
                raise SystemExit(f"{name} {t}: nearest valid_time is off by {gap}")
            vals.append(float(ds.dis06.isel(step=si, time=ti, y=yi, x=xi).values))
        out[name] = vals
    ds.close()
    return out


def main():
    if not EFAS_DIR.exists():
        raise SystemExit(f"EFAS source directory not found: {EFAS_DIR}")

    frame = pd.DataFrame({"idx": range(NSTEP), "datetime": TARGET})
    for key, fn in SOURCES.items():
        got = extract(EFAS_DIR / fn)
        for st, vals in got.items():
            frame[f"{st}_{key}"] = np.round(vals, 4)

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "efas_station_Q.csv"
    frame.to_csv(dest, index=False)
    print(f"wrote {dest}  ({NSTEP} steps, {len(SOURCES) * len(STATIONS)} series)")

    if "--verify" in sys.argv:
        src = Path("/Users/haseeb/Documents/Phd_thesis/Research_papers/"
                   "WRF_LISFLOOD_Research_paper_v1/flood_simulations_csv")
        print("\nagainst the published series, joined on valid time:")
        worst, compared = 0.0, 0
        new = frame.set_index("datetime")
        for st in STATIONS:
            p = src / f"{st.lower()}_timeseries.csv"
            if not p.exists():
                continue
            old = pd.read_csv(p)
            old["Time"] = pd.to_datetime(old["Time"])
            old = old.set_index("Time")
            # Join on the timestamp, never on row position: the two series now start
            # at different epochs, so aligning by position would silently compare
            # different times and report a spurious agreement.
            common = new.index.intersection(old.index)
            if len(common) == 0:
                print(f"  {st:12s} no overlapping timestamps with the published series")
                continue
            for key in SOURCES:
                d = new.loc[common, f"{st}_{key}"].values - old.loc[common, key].values
                worst = max(worst, float(np.abs(d).max()))
                compared += len(common)
                print(f"  {st:12s} {key:11s} {len(common):2d} common steps, "
                      f"max |diff| = {np.abs(d).max():.6f} m3/s")
        print(f"\nlargest disagreement over {compared} matched values: {worst:.6f} m3/s")


if __name__ == "__main__":
    main()
