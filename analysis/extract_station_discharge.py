#!/usr/bin/env python3
"""
extract_station_discharge.py
Extract per-station discharge & depth from LISFLOOD-FP 6-hourly Qx/Qy/wd grids,
and the fine-resolution outlet discharge (Qout) from the mass file, for each
forcing run (wrf, wrf12, graphcast, fuxi, aifs). Writes small CSVs.
"""
import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import rasterio
from datetime import datetime, timedelta

RIVER_WIDTH = 20.0
CELL_SIZE = 10.0
START = datetime(2021, 7, 13, 0, 0)
INTERVAL_H = 6
STATIONS = {
    "Walferdange": {"x": 77256, "y": 81571, "type": "water_level"},
    "Steinsel":    {"x": 77432, "y": 82659, "type": "discharge"},
    "Pfaffenthal": {"x": 77409, "y": 76226, "type": "discharge"},
    "Livange":     {"x": 76151, "y": 65753, "type": "discharge"},
    "Hesperange":  {"x": 78623, "y": 72404, "type": "discharge"},
}


def _ts(fn):
    m = re.search(r"-(\d+)", fn)
    return int(m.group(1)) if m else 0


def extract_run(results_dir):
    qxs = sorted(glob.glob(os.path.join(results_dir, "*.Qx")),
                 key=lambda p: _ts(os.path.basename(p)))
    if not qxs:
        return None
    with rasterio.open(qxs[0]) as s:
        meta = {n: s.index(i["x"], i["y"]) for n, i in STATIONS.items()}
        shape = s.shape
    rows = []
    for idx, qx in enumerate(qxs):
        qy, wd = qx.replace(".Qx", ".Qy"), qx.replace(".Qx", ".wd")
        if not (os.path.exists(qy) and os.path.exists(wd)):
            continue
        with rasterio.open(qx) as a, rasterio.open(qy) as b, rasterio.open(wd) as c:
            QX, QY, WD, nod = a.read(1), b.read(1), c.read(1), c.nodata
        rec = {"Time": START + timedelta(hours=idx * INTERVAL_H)}
        for n, (r, cc) in meta.items():
            qxc, qyc = QX[r, cc], QY[r, cc]
            fm = float(np.hypot(qxc, qyc))
            px, py = (-qyc / fm, qxc / fm) if fm > 1e-3 else (1.0, 0.0)
            hw, ns, Q = RIVER_WIDTH / 2.0, int(RIVER_WIDTH / CELL_SIZE) + 1, 0.0
            for i in range(ns):
                d = -hw + i * CELL_SIZE
                sr, sc = int(r + d * py / CELL_SIZE), int(cc + d * px / CELL_SIZE)
                if 0 <= sr < shape[0] and 0 <= sc < shape[1]:
                    Q += float(np.hypot(QX[sr, sc], QY[sr, sc])) * CELL_SIZE
            dep = WD[r, cc]
            dep = 0.0 if (dep < 0 or dep == nod) else float(dep)
            rec[f"{n}_Q"] = Q
            rec[f"{n}_Depth"] = dep
        rows.append(rec)
    return pd.DataFrame(rows)


def read_mass(path):
    cols = ["Time", "Tstep", "MinTstep", "NumTsteps", "Area", "Vol",
            "Qin", "Hds", "Qout", "Qerror", "Verror", "RainInfEvap"]
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, names=cols, engine="python")
    return df[["Time", "Qout"]]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for M in ["wrf", "wrf12", "graphcast", "fuxi", "aifs"]:
        rd = os.path.join(root, "runs", M, "results")
        df = extract_run(rd)
        if df is not None and len(df):
            out = os.path.join(rd, f"{M}_station_Q.csv")
            df.to_csv(out, index=False)
            peaks = {n: round(df[f"{n}_Q"].max(), 1) for n in STATIONS}
            print(f"{M}: {len(df)} steps -> {out}  peak Q {peaks}")
        else:
            print(f"{M}: no Qx grids found in {rd}")
        mp = os.path.join(rd, "6hr.mass")
        if os.path.exists(mp):
            m = read_mass(mp)
            m.to_csv(os.path.join(rd, f"{M}_outlet_Qout.csv"), index=False)
            print(f"  {M} outlet: {len(m)} rows, peak Qout = {m.Qout.max():.1f} m3/s")


if __name__ == "__main__":
    main()
