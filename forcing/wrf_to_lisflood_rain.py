#!/usr/bin/env python3
"""
wrf_to_lisflood_rain.py  (102 h window)
Convert WRF wrfout 6-hourly TOTAL precipitation (RAINC+RAINSH+RAINNC) into a
LISFLOOD-FP dynamicrainfile NetCDF on the 10 m Alzette grid (EPSG:2169), byte-
compatible with rain_10m.nc. Works for any After-DA domain (d01=12 km, d03=1.3 km).

Each wrfout holds the per-cycle 6 h accumulation (rapid-update cycle -> NO
differencing). At convection-permitting d03, RAINC~0 so total==RAINNC and this
reproduces the existing rain_10m.nc; at d01 (12 km, cumulus scheme) RAINC>0, so
"total" is required to include convective rain.

Usage: python wrf_to_lisflood_rain.py <wrf_dir> <domain: d01|d03> <dem_asc> <out_nc>
Window 2021-07-13 00:00 -> 17 06:00 (102 h, 6-hourly => 18 steps incl. zero at t=0).
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import griddata

START = pd.Timestamp("2021-07-13 00:00")
END = pd.Timestamp("2021-07-17 06:00")
STEP_H = 6
PROJ4_2169 = ("+proj=tmerc +lat_0=49.8333333333333 +lon_0=6.16666666666667 "
              "+k=0.9996 +x_0=80000 +y_0=100000 +ellps=GRS80 +units=m +no_defs")


def read_asc_grid(path):
    hdr = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
    ncols, nrows = int(hdr["ncols"]), int(hdr["nrows"])
    xll, yll, cs = hdr["xllcorner"], hdr["yllcorner"], hdr["cellsize"]
    x = xll + (np.arange(ncols) + 0.5) * cs
    y = (yll + (np.arange(nrows) + 0.5) * cs)[::-1]   # DESCENDING (north first)
    return x, y


def main():
    wdir, dom, dem, out = sys.argv[1:5]
    x, y = read_asc_grid(dem)
    nx, ny = x.size, y.size
    Xt, Yt = np.meshgrid(x, y)
    target = np.column_stack([Xt.ravel(), Yt.ravel()])
    tr = Transformer.from_crs("EPSG:4326", "EPSG:2169", always_xy=True)
    vts = pd.date_range(START + pd.Timedelta(hours=STEP_H), END, freq=f"{STEP_H}h")
    layers = []
    for t in vts:
        f = os.path.join(wdir, f"wrfout_{dom}_{t.strftime('%Y-%m-%d_%H')}_00_00")
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        ds = xr.open_dataset(f)
        p = (ds["RAINNC"] + ds["RAINC"] + ds["RAINSH"]).isel(Time=0).values
        lon = ds["XLONG"].isel(Time=0).values.ravel()
        lat = ds["XLAT"].isel(Time=0).values.ravel()
        ds.close()
        pm = np.asarray(p).ravel()
        g = np.isfinite(pm)
        xs, ys = tr.transform(lon[g], lat[g])
        grid = griddata((xs, ys), pm[g], target, method="linear", fill_value=0.0).reshape(ny, nx)
        grid = np.clip(grid, 0.0, None).astype(np.float32)
        layers.append(grid)
        print(f"  {dom} {t.strftime('%m-%d_%H')}: src_max={pm.max():6.1f}mm grid_mean={grid.mean():6.2f}mm", flush=True)
    rain = np.concatenate([np.zeros((1, ny, nx), np.float32), np.stack(layers)], axis=0)
    th = (np.arange(rain.shape[0]) * STEP_H).astype(np.float64)
    print(f"  EVENT total domain-mean = {rain.mean(axis=(1, 2)).sum():.1f} mm")
    od = xr.Dataset(
        {
            "time": ("time", th, {"units": "hour", "long_name": "time", "axis": "T"}),
            "x": ("x", x, {"units": "m", "long_name": "x", "axis": "X", "spatial_ref": "EPSG:2169"}),
            "y": ("y", y, {"units": "m", "long_name": "y", "axis": "Y", "spatial_ref": "EPSG:2169"}),
            "rainfall_depth": (("time", "y", "x"), rain, {"units": "mm", "long_name": "rainfall_depth"}),
        },
        attrs={"crs": "EPSG:2169", "spatial_ref": PROJ4_2169, "source": f"WRF {dom} total precip"},
    )
    od.to_netcdf(out, format="NETCDF4",
                 encoding={"time": {"_FillValue": None, "dtype": "float64"},
                           "x": {"_FillValue": None, "dtype": "float64"},
                           "y": {"_FillValue": None, "dtype": "float64"},
                           "rainfall_depth": {"_FillValue": None, "dtype": "float32", "zlib": True, "complevel": 4}},
                 unlimited_dims=["time"])
    print("WROTE", out, rain.shape)


if __name__ == "__main__":
    main()
