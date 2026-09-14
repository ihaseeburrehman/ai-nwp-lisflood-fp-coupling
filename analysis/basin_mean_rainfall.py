#!/usr/bin/env python3
"""Basin-mean vs domain-mean event rainfall for each LISFLOOD-FP forcing.

Table 1 of the manuscript reports a mean over the whole 10 m hydraulic domain,
a rectangle of about 1064 km2 that extends well beyond the Alzette sub-basin.
This script recomputes the same quantities over the basin itself (467.2 km2,
Alzette_sub_basin_complete.shp) so the two can be compared, and prints the
in-basin rain-gauge totals alongside.

Reads:
  data/lisflood_96h/rain_18utc/rain_*_96h.nc          the LISFLOOD rain inputs
  <SSD>/.../Alzette_sub_basin_complete.shp             basin outline, EPSG:2169
  <GMD>/.../stations_6hr_cumulative_expanded.xlsx      AgriMeteo gauge records
Writes:
  data/lisflood_96h/rainfall_domain_vs_basin.csv

    python3 scripts/basin_mean_rainfall.py
"""
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
RAIN = ROOT / "data" / "lisflood_96h" / "rain_18utc"
OUT = ROOT / "data" / "lisflood_96h" / "rainfall_domain_vs_basin.csv"

BASIN = ("/Volumes/SanDisk 2TB/2026-07-14-115148.previous/Data/Users/haseeb.rehman/"
         "Documents/Misc/Lisflood_Simulations/Lisflood_Alzette_river_basin/sub_basins/"
         "5m/sub_basin_complete/pre_processing/Alzette_sub_basin_complete.shp")
GAUGES = ("/Users/haseeb/Documents/Phd_thesis/Research_papers/WRF_vs_AI_GMD/03_Revised_rv2/"
          "analysis/data/station_observations/stations_6hr_cumulative_expanded.xlsx")

FORCINGS = [
    ("WRF 1.3 km (After-DA)",  "rain_wrf1p3_afterda_96h.nc"),
    ("WRF 12 km (After-DA)",   "rain_wrf12_afterda_96h.nc"),
    ("WRF 12 km (Before-DA)",  "rain_wrf12_beforeda_96h.nc"),
    ("AIFS",                   "rain_aifs_96h.nc"),
    ("GraphCast",              "rain_graphcast_v2_96h.nc"),
    ("FuXi",                   "rain_fuxi_96h.nc"),
]
# Run epoch: the start grid 13_july_18hr.wd is the model state at 13 July 18:00 UTC.
T0 = pd.Timestamp("2021-07-13 18:00")
T_END = pd.Timestamp("2021-07-17 18:00")

# Flood-generating burst, defined as absolute times and converted to hours from T0
# so it cannot drift if the epoch changes again.
_B0, _B1 = pd.Timestamp("2021-07-14 12:00"), pd.Timestamp("2021-07-15 00:00")
BURST = ((_B0 - T0).total_seconds() / 3600.0,
         (_B1 - T0).total_seconds() / 3600.0)


MASK_CACHE = ROOT / "data" / "lisflood_96h" / "basin_mask.npy"


def basin_mask(ds):
    """Boolean array, True inside the basin, on the rain grid.

    The basin outline lives on an external drive, which makes this script
    unrunnable whenever that drive is absent. The rasterised mask is therefore
    cached next to the data the first time it is built, and reused thereafter, so
    the shapefile is needed exactly once and the released code reproduces without
    it.
    """
    if MASK_CACHE.exists():
        m = np.load(MASK_CACHE)
        if m.shape == (len(ds.y), len(ds.x)):
            return m
        raise SystemExit(f"{MASK_CACHE} has shape {m.shape}, expected "
                         f"{(len(ds.y), len(ds.x))}; delete it to rebuild.")

    if not os.path.exists(BASIN):
        raise SystemExit(
            f"basin outline not found:\n  {BASIN}\n"
            f"and no cached mask at\n  {MASK_CACHE}\n"
            "Mount the drive holding the shapefile once; the mask will then be "
            "cached and the drive is not needed again.")

    x, y = ds.x.values, ds.y.values
    dx = float(abs(x[1] - x[0]))
    dy = float(abs(y[1] - y[0]))
    # y descends, so the grid origin is the north-west corner
    tr = from_origin(x[0] - dx / 2, y[0] + dy / 2, dx, dy)
    g = gpd.read_file(BASIN).to_crs(ds.attrs["crs"])
    inside = ~geometry_mask(g.geometry, out_shape=(len(y), len(x)),
                            transform=tr, invert=False, all_touched=False)
    MASK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(MASK_CACHE, inside)
    print(f"cached basin mask -> {MASK_CACHE}")
    return inside


def stats(series, hours):
    tot = float(series.sum())
    k = int(np.argmax(series))
    # hours are interval-END hours (see stamp_convention below), so the 12 h burst
    # is the two accumulations ending at BURST[0]+6 and BURST[1]: lower bound exclusive.
    burst = float(sum(v for h, v in zip(hours, series) if BURST[0] < h <= BURST[1]))
    return dict(total=tot, peak6=float(series[k]),
                peak_time=(T0 + pd.Timedelta(hours=float(hours[k]))).strftime("%d %b %H"),
                burst_pct=100.0 * burst / tot if tot else np.nan)


rows, mask, ncell = [], None, None
for label, fn in FORCINGS:
    with xr.open_dataset(RAIN / fn) as ds:
        if mask is None:
            mask = basin_mask(ds)
            ncell = int(mask.sum())
            print(f"basin mask: {ncell:,} cells of {mask.size:,} "
                  f"= {ncell * 1e-4:.1f} km2 of {mask.size * 1e-4:.1f} km2\n")
        # Both the stamping convention and the absolute epoch must be declared by
        # the file: inferring either one lets an old-epoch file be read against a
        # new-epoch window, which displaces every result by six hours without any
        # visible symptom. Nothing is assumed here.
        conv = ds.attrs.get("time_stamp_convention")
        w0 = ds.attrs.get("window_start_utc")
        if conv is None or w0 is None:
            raise SystemExit(
                f"{label}: {fn} does not declare time_stamp_convention and "
                "window_start_utc. Files built before 2026-09-09 lack them and "
                "belong to the old 13 Jul 00:00 window; point RAIN at the "
                "corrected forcing rather than mixing epochs.")
        if conv not in ("interval_start", "interval_end"):
            raise SystemExit(f"{label}: unknown time_stamp_convention {conv!r}")
        if pd.Timestamp(w0.replace("Z", "")) != T0:
            raise SystemExit(
                f"{label}: {fn} declares window_start_utc {w0}, but this script is "
                f"configured for {T0:%Y-%m-%d %H:%M}. Refusing to mix epochs.")
        hours = ds.time.values.astype(float)
        if conv == "interval_start":
            hours = hours + 6.0
        r = ds["rainfall_depth"]
        dom = r.mean(dim=("y", "x")).values.astype(float)
        bas = r.where(xr.DataArray(mask, dims=("y", "x"))).mean(dim=("y", "x")).values.astype(float)
    d, b = stats(dom, hours), stats(bas, hours)
    rows.append(dict(forcing=label, domain_total=d["total"], basin_total=b["total"],
                     domain_peak6=d["peak6"], basin_peak6=b["peak6"],
                     peak_time=b["peak_time"], basin_burst_pct=b["burst_pct"]))
    print(f"{label:<24} domain {d['total']:6.1f} mm  basin {b['total']:6.1f} mm  "
          f"({100 * (b['total'] / d['total'] - 1):+5.1f}%)   "
          f"peak6 {d['peak6']:5.1f} -> {b['peak6']:5.1f}   {b['peak_time']}")

T = pd.DataFrame(rows)
T.to_csv(OUT, index=False, float_format="%.2f")

book = pd.ExcelFile(GAUGES)
gauge_totals = {}
print()
for st in ["Livange", "Oberkorn"]:
    g = pd.read_excel(book, sheet_name=st)
    g.columns = [str(c).strip() for c in g.columns]
    s = pd.Series(pd.to_numeric(g["Precip(mm)"], errors="coerce").values,
                  index=pd.to_datetime(g["UTC_Datetime"], errors="coerce")).dropna()
    # Gauge records are 6-hourly accumulations labelled at the END of their interval,
    # so the value stamped exactly at T0 covers the six hours BEFORE the window and
    # must be excluded: the lower bound is exclusive, the upper inclusive.
    w = s.loc[(s.index > T0) & (s.index <= T_END)]
    gauge_totals[st] = float(w.sum())
    print(f"gauge {st:<18} total {w.sum():6.1f} mm  peak6 {w.max():5.1f} mm  "
          f"{w.idxmax():%d %b %H}")

ref = T.set_index("forcing").loc["WRF 1.3 km (After-DA)"]
gmean = sum(gauge_totals.values()) / len(gauge_totals)
print(f"\nWRF 1.3 km basin mean {ref.basin_total:.1f} mm against a two-gauge mean of "
      f"{gmean:.1f} mm: {100 * (ref.basin_total / gmean - 1):+.0f}%")
print(f"wrote {OUT}")
