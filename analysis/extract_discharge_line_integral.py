#!/usr/bin/env python3
"""Discharge at each gauge as a line integral across the surveyed section.

This is the method used for the reported results. It replaces the nearest-cell
sampling of the earlier extractors, which was not rounding-independent: with
only two to six sample points across a 20-50 m section, and a steep flux
gradient between channel and bank, truncating rather than rounding the cell
index changed the peak at Steinsel by 74 per cent. That ambiguity is removed
here by evaluating

    Q(t) = integral over Gamma of ( q . n ) ds

exactly: the transect is traversed cell by cell and each crossed cell contributes
its exact traversed length, so no quadrature step and no rounding convention enters
the result. The script instead reports how much the answer moves when the section is
slid along the reach by sub-cell amounts, which is the sensitivity that does remain.

Three further choices, all stated in the manuscript:
  * the section width is the river width surveyed at each gauge from winter
    orthoimagery (Sect. 3.6);
  * the section normal n is the downstream direction of the river centreline,
    not the local (Qx, Qy), which is unreliable where the flow is out of bank;
  * the flux is signed, q . n, so water moving across or against the reach is
    not counted as downstream discharge.

Reads:
  data/lisflood_96h/runs/<model>/results/6hr-*.{Qx,Qy,wd}
  <SSD>/.../alzette_river.shp
Writes:
  data/lisflood_96h/line_integral_18utc/<model>_station_Q.csv
  data/lisflood_96h/line_integral_18utc/placement_sensitivity.csv

    python3 scripts/extract_discharge_line_integral.py [model ...]
"""
import re
import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from shapely.ops import nearest_points

ROOT = Path(__file__).resolve().parents[1]
LF = ROOT / "data" / "lisflood_96h"
RUNS = LF / "runs"
OUT = LF / "line_integral_18utc"
RIVER = ("/Volumes/SanDisk 2TB/2026-07-14-115148.previous/Data/Users/haseeb.rehman/"
         "Documents/Misc/Lisflood_Simulations/Lisflood_Alzette_river_basin/sub_basins/"
         "5m/sub_basin_complete/pre_processing/alzette_river.shp")

CELL = 10.0
# Option B window: the start grid 13_july_18hr.wd is the model state at
# 13 July 18:00 UTC, so the run clock and the forcing both begin there.
START = datetime(2021, 7, 13, 18, 0)
INTERVAL_H = 6
TANGENT_M = 100.0

# Gauge coordinates (EPSG:2169) and the surveyed river width at each.
# Pfaffenthal is 30 m rather than 20: the DEM shows a hard wall 20 m west of the
# thalweg but a shelf 20 m east that is 0.27 m below the flood water surface and
# does convey water, so the section is asymmetric and wider than the visible
# low-flow channel.
STATIONS = {
    "Walferdange": (77256, 81571, 30.0),
    "Steinsel":    (77432, 82659, 50.0),
    "Pfaffenthal": (77409, 76226, 30.0),
    "Livange":     (76151, 65753, 25.0),
    "Hesperange":  (78623, 72404, 25.0),
}
# The six forcings reported in the manuscript. An ECMWF HRES-forced run also
# exists on the HPC but is deliberately excluded from the study, so it is not
# extracted and not resubmitted.
MODELS = ["wrf1p3_afterda", "wrf12_afterda", "wrf12_beforeda",
          "graphcast", "aifs", "fuxi"]


def header(p):
    h = {}
    with open(p) as f:
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
    h["ncols"], h["nrows"] = int(h["ncols"]), int(h["nrows"])
    return h


def grid(p, h):
    return np.loadtxt(p, skiprows=6, dtype=np.float32).reshape(h["nrows"], h["ncols"])


def centreline_dirs(bed=None, h=None):
    """Downstream unit vector at each gauge, from the river centreline.

    The chord is taken along the single connected part nearest the gauge, not
    along the unioned geometry: this shapefile is a MultiLineString whose second
    part is a 1 m fragment, and a chord spanning parts would be meaningless.
    Shapely's parameterisation happens to run downstream here, but that is not
    guaranteed, so when a bed-elevation grid is supplied the sense is checked
    against it -- sampled ALONG the centreline, since a straight offset lands on
    the bank at a meander -- and reversed if the bed rises downstream.
    """
    riv = gpd.read_file(RIVER).to_crs(2169)
    geom = riv.geometry.union_all() if hasattr(riv.geometry, "union_all") \
        else riv.geometry.unary_union
    parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
    out = {}
    for n, (x, y, _) in STATIONS.items():
        p = Point(x, y)
        part = min(parts, key=lambda g: g.distance(p))
        s0 = part.project(nearest_points(part, p)[0])
        a = part.interpolate(max(s0 - TANGENT_M / 2, 0.0))
        b = part.interpolate(min(s0 + TANGENT_M / 2, part.length))
        dx, dy = b.x - a.x, b.y - a.y
        m = np.hypot(dx, dy)
        if m < 1e-6:
            raise SystemExit(f"degenerate centreline chord at {n}")
        ux, uy = dx / m, dy / m
        if bed is not None:
            L = 300.0
            pa = part.interpolate(max(s0 - L, 0.0))
            pb = part.interpolate(min(s0 + L, part.length))
            top = h["yllcorner"] + h["nrows"] * h["cellsize"]
            def _bed(pt):
                cc = int(np.floor((pt.x - h["xllcorner"]) / h["cellsize"]))
                rr = int(np.floor((top - pt.y) / h["cellsize"]))
                return float(bed[rr, cc])
            if _bed(pb) > _bed(pa):          # bed rises downstream -> reversed
                ux, uy = -ux, -uy
                print(f"  {n}: centreline sense reversed to run downhill")
        out[n] = (ux, uy)
    return out


def transect_cells(h, x, y, W, px, py):
    """Cells the transect crosses, with the EXACT length of transect in each.

    A grid traversal rather than a sampled approximation: the parameter values
    at which the transect crosses each cell boundary are solved directly, so the
    length assigned to every cell is exact and no quadrature error remains.
    Replaces an earlier midpoint Riemann sum whose error at 0.5 m was still
    1.1 per cent at two gauges.
    """
    c = h["cellsize"]
    top = h["yllcorner"] + h["nrows"] * c
    t0, t1 = -W / 2.0, W / 2.0
    breaks = {t0, t1}
    if abs(px) > 1e-12:
        ka = int(np.floor(min(x + t0 * px, x + t1 * px) / c - h["xllcorner"] / c))
        kb = int(np.ceil(max(x + t0 * px, x + t1 * px) / c - h["xllcorner"] / c))
        for k in range(ka, kb + 1):
            t = (h["xllcorner"] + k * c - x) / px
            if t0 < t < t1:
                breaks.add(float(t))
    if abs(py) > 1e-12:
        ka = int(np.floor((top - max(y + t0 * py, y + t1 * py)) / c))
        kb = int(np.ceil((top - min(y + t0 * py, y + t1 * py)) / c))
        for k in range(ka, kb + 1):
            t = (top - k * c - y) / py
            if t0 < t < t1:
                breaks.add(float(t))
    ts = sorted(breaks)
    out = []
    for ta, tb in zip(ts[:-1], ts[1:]):
        L = tb - ta
        if L <= 1e-12:
            continue
        tm = 0.5 * (ta + tb)
        cc = int(np.floor((x + tm * px - h["xllcorner"]) / c))
        rr = int(np.floor((top - (y + tm * py)) / c))
        out.append((rr, cc, L))
    return out


def line_integral(QX, QY, h, x, y, W, ux, uy, ds=None):
    """Signed normal flux across a transect of width W centred on (x, y).

    Q = sum over crossed cells of (q . n) * (exact transect length in the cell),
    in m3/s for Qx, Qy in m2/s. `ds` is accepted for call compatibility and ignored:
    the quadrature is exact, so there is no step size to converge.
    A section that is not fully covered raises rather than returning a partial
    discharge.
    """
    px, py = -uy, ux
    nodata = h.get("nodata_value", -9999.0)
    Q, covered = 0.0, 0.0
    for rr, cc, L in transect_cells(h, x, y, W, px, py):
        if not (0 <= rr < h["nrows"] and 0 <= cc < h["ncols"]):
            continue
        qx, qy = float(QX[rr, cc]), float(QY[rr, cc])
        if qx == nodata or qy == nodata:
            continue
        Q += (qx * ux + qy * uy) * L
        covered += L
    if covered < W - 1e-6:
        raise SystemExit(f"transect at ({x:.0f},{y:.0f}) covers {covered:.2f} m of "
                         f"{W:.2f} m; outside the grid or masked")
    return Q


def par_for(model):
    """Locate the .par that produced a run, if it sits beside the results."""
    for cand in (RUNS.parent / f"{model}.par", RUNS / model / f"{model}.par"):
        if cand.exists():
            return cand
    return None


def run_provenance(model, rd, nsteps):
    """Record what a run was actually made of, and refuse a mismatched epoch.

    The absolute start time of a LISFLOOD-FP run exists nowhere in its own
    configuration: sim_time is a duration and the rain axis is relative hours. So
    the epoch this extractor stamps onto every row is an assumption, and if the
    grids came from a run forced by a different window the CSVs are silently wrong.
    Where the .par is available the forcing file is read and its declared window
    checked against START; the result is written next to the CSVs so the
    assumption travels with the data instead of living only in this file.
    """
    rec = {"model": model, "results_dir": str(rd), "steps": nsteps,
           "assumed_start_utc": START.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "interval_h": INTERVAL_H}
    par = par_for(model)
    if par is None:
        rec["par"] = None
        rec["epoch_verified"] = False
        rec["note"] = ("no .par found beside the results; the run epoch could not be "
                       "verified against the forcing and is assumed")
        return rec
    rec["par"] = str(par)
    cfg = {}
    for line in par.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.startswith("#"):
            cfg.setdefault(parts[0], parts[1])
    rec["rainfile"] = cfg.get("dynamicrainfile")
    rec["startfile"] = cfg.get("startfile")
    rec["sim_time_s"] = cfg.get("sim_time")
    rf = cfg.get("dynamicrainfile")
    if rf and os.path.exists(rf):
        with xr.open_dataset(rf) as ds:
            w0 = ds.attrs.get("window_start_utc")
            rec["forcing_window_start_utc"] = w0
            rec["forcing_convention"] = ds.attrs.get("time_stamp_convention")
        if w0 is None:
            raise SystemExit(
                f"{model}: forcing {os.path.basename(rf)} does not declare "
                "window_start_utc, so the run epoch cannot be verified. Rebuild the "
                "forcing with the corrected post-processor before extracting.")
        if pd.Timestamp(w0.replace("Z", "")) != pd.Timestamp(START):
            raise SystemExit(
                f"{model}: forcing declares window_start_utc {w0} but this extractor "
                f"stamps {START:%Y-%m-%d %H:%M}. Refusing to write CSVs that would "
                "carry the wrong timestamps.")
        rec["epoch_verified"] = True
    else:
        rec["epoch_verified"] = False
        rec["note"] = "forcing file named in the .par is not readable from here"
    return rec


def main():
    want = sys.argv[1:] or MODELS
    OUT.mkdir(parents=True, exist_ok=True)
    conv = []
    manifest = []
    for model in want:
        rd = RUNS / model / "results"
        qxs = sorted(rd.glob("*.Qx"),
                     key=lambda p: int(re.search(r"-(\d+)", p.name).group(1)))
        if not qxs:
            print(f"{model}: no Qx grids"); continue
        h = header(qxs[0])
        bed = np.load(LF / "bed_elevation.npy") if (LF / "bed_elevation.npy").exists() else None
        dirs = centreline_dirs(bed, h)
        top = h["yllcorner"] + h["nrows"] * h["cellsize"]
        rows = []
        for qx in qxs:
            idx = int(re.search(r"-(\d+)", qx.name).group(1))   # from the filename
            qy = Path(str(qx).replace(".Qx", ".Qy"))
            wd = Path(str(qx).replace(".Qx", ".wd"))
            if not (qy.exists() and wd.exists()):
                print(f"  {model}: step {idx} incomplete, skipped")
                continue
            QX, QY, WD = grid(qx, h), grid(qy, h), grid(wd, h)
            rec = {"Time": START + timedelta(hours=idx * INTERVAL_H)}
            for n, (x, y, W) in STATIONS.items():
                ux, uy = dirs[n]
                rec[f"{n}_Q"] = line_integral(QX, QY, h, x, y, W, ux, uy)
                cc = int(np.floor((x - h["xllcorner"]) / h["cellsize"]))
                rr = int(np.floor((top - y) / h["cellsize"]))
                # depth in the single gauge cell, not a section mean or a stage
                rec[f"{n}_Depth_cell"] = max(float(WD[rr, cc]), 0.0)
                if model == want[0]:
                    # Placement sensitivity. The quadrature is exact, so varying a step
                    # size proves nothing; what the result can still depend on is where
                    # the section sits relative to the 10 m cells. Slide the section
                    # along the reach by sub-cell amounts and report the spread. This is
                    # evaluated at each station's OWN peak step, selected after all steps
                    # have been read, because the stations do not peak simultaneously.
                    for off in (-5.0, -2.5, 0.0, 2.5, 5.0):
                        conv.append(dict(
                            station=n, step=idx, offset_m=off,
                            Q=line_integral(QX, QY, h, x + off * ux, y + off * uy,
                                            W, ux, uy)))
            rows.append(rec)
        d = pd.DataFrame(rows)
        manifest.append(run_provenance(model, rd, len(d)))
        d.to_csv(OUT / f"{model}_station_Q.csv", index=False, float_format="%.4f")
        print(f"{model}: {len(d)} steps   peak Q "
              + str({n: round(float(d[f'{n}_Q'].max()), 1) for n in STATIONS}))
    if manifest:
        mpath = OUT / "run_manifest.json"
        mpath.write_text(json.dumps(
            {"extracted_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
             "assumed_start_utc": START.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "runs": manifest}, indent=2))
        nver = sum(1 for r in manifest if r.get("epoch_verified"))
        print(f"\nwrote {mpath}  ({nver}/{len(manifest)} runs epoch-verified "
              "against their forcing)")
    if conv:
        cv = pd.DataFrame(conv)
        # each station's peak step, taken from the unshifted section
        pk = (cv[cv.offset_m == 0.0]
              .loc[lambda t: t.groupby("station")["Q"].idxmax(), ["station", "step"]])
        cv = cv.merge(pk, on=["station", "step"])
        c = cv.pivot(index="station", columns="offset_m", values="Q")
        c["spread_pct"] = 100.0 * (c.max(axis=1) - c.min(axis=1)) / c[0.0]
        c.to_csv(OUT / "placement_sensitivity.csv", float_format="%.3f")
        print("\nsensitivity to section placement along the reach "
              "(m3/s at each station's own peak step, offsets in m):")
        print(c.round(2).to_string())


if __name__ == "__main__":
    main()
