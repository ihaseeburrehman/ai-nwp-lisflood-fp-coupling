#!/usr/bin/env python3
"""
catchment_hyetograph.py
Catchment-mean rainfall diagnostics for the five LISFLOOD-FP forcings.

Reads the dynamicrainfile NetCDFs written by code_release/forcing/*_to_lisflood_rain.py
(variable `rainfall_depth`, dims (time, y, x), 6-hourly accumulations in mm, first
layer zero) and produces:

  1. data/pgfplots/hyetograph.csv  -- catchment-mean 6-h rainfall per forcing,
     in the same idx/column layout as the *_merged.csv files the manuscript's
     pgfplots figures already read.
  2. data/pgfplots/forcing_intensity.csv -- the per-forcing summary numbers for
     the extra Table 1 columns: event total, peak 6-h catchment-mean intensity,
     and the fraction of the event total falling in the critical 24-48 h window.

The domain mean reproduced here is the same quantity Table 1 already reports, so
the WRF 1.3 km event total is a built-in check: it should come back as 87.4 mm.

Usage:
    python catchment_hyetograph.py [<rain_nc_dir>] [<out_dir>]

<rain_nc_dir> defaults to data/lisflood_96h/rain_18utc and must contain the six
NetCDFs rain_<stem>_96h.nc listed in FORCINGS below.
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

# label -> filename stem, in the manuscript's plotting order
FORCINGS = [
    ("WRF_1p3km", "wrf1p3_afterda"),
    ("WRF_12km", "wrf12_afterda"),
    ("WRF_12km_BeforeDA", "wrf12_beforeda"),
    ("GraphCast", "graphcast_v2"),
    ("FuXi", "fuxi"),
    ("AIFS", "aifs"),
]

START = pd.Timestamp("2021-07-13 18:00")   # Option B: run epoch = start-grid time
STEP_H = 6
# critical window for the Alzette response, measured from the start of the
# 96-h flood bracket; matches the 24-48 h window discussed in the manuscript.
# Windows are defined as ABSOLUTE times and converted to hours from START, so
# they cannot drift if the run epoch changes again.
CRIT_START, CRIT_END = pd.Timestamp('2021-07-14 00:00'), pd.Timestamp('2021-07-15 00:00')
# Flood-generating burst reported in the 'Burst share' column of Table 1:
# 14 July 12:00 - 15 July 00:00 UTC. Each value is the accumulation ENDING at its
# timestamp, so the 12 h from +36 h to +48 h are the two accumulations stamped 42
# and 48 h; the lower bound is therefore exclusive. (Summing 36, 42 and 48 would
# span +30 to +48 h, i.e. 18 hours, which is what earlier versions reported.)
BURST_START, BURST_END = pd.Timestamp('2021-07-14 12:00'), pd.Timestamp('2021-07-15 00:00')
CRIT_START_H = (CRIT_START - START).total_seconds() / 3600.0
CRIT_END_H = (CRIT_END - START).total_seconds() / 3600.0
BURST_START_H = (BURST_START - START).total_seconds() / 3600.0
BURST_END_H = (BURST_END - START).total_seconds() / 3600.0


def file_epoch_and_convention(path, label):
    """Return (convention, window_start) declared BY THE FILE, or abort.

    Two independent things can be wrong and neither is visible in the data:
    the stamping convention (does a value cover [t, t+6h) or (t-6h, t]?) and the
    absolute epoch (what wall-clock time is t = 0?). Reading an old midnight-epoch
    file while the script is configured for the 18:00 epoch silently displaces
    every window by six hours and yields plausible, wrong burst shares. So both
    are required to be declared in the file and are checked against the
    configuration here; nothing is inferred.
    """
    with xr.open_dataset(path) as ds:
        conv = ds.attrs.get("time_stamp_convention")
        w0 = ds.attrs.get("window_start_utc")
    if conv is None or w0 is None:
        raise SystemExit(
            f"{label}: {os.path.basename(path)} does not declare "
            "time_stamp_convention and window_start_utc.\n"
            "Files built before 2026-09-09 lack them and belong to the old "
            "13 Jul 00:00 window. Point RAIN at the corrected forcing, or restore "
            "the matching epoch, rather than mixing the two.")
    if conv not in ("interval_start", "interval_end"):
        raise SystemExit(f"{label}: unknown time_stamp_convention {conv!r}")
    declared = pd.Timestamp(w0.replace("Z", ""))
    if declared != START:
        raise SystemExit(
            f"{label}: {os.path.basename(path)} declares window_start_utc "
            f"{declared:%Y-%m-%d %H:%M}, but this script is configured for "
            f"{START:%Y-%m-%d %H:%M}. Refusing to mix epochs.")
    return conv, declared


def catchment_mean_series(path):
    """Return (hours, catchment-mean 6-h rainfall in mm) for one forcing."""
    with xr.open_dataset(path) as ds:
        rain = ds["rainfall_depth"]
        # spatial mean over the whole 10 m hydraulic domain -- the same
        # quantity as the 'catchment-mean event total' of Table 1
        series = rain.mean(dim=("y", "x")).values.astype(float)
        hours = ds["time"].values.astype(float)
    return hours, series


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    rdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        root, "data", "lisflood_96h", "rain_18utc")
    odir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "data", "pgfplots")
    os.makedirs(odir, exist_ok=True)

    hyeto, summary, ref_hours, ref_hours_end = {}, [], None, None

    for label, stem in FORCINGS:
        path = os.path.join(rdir, f"rain_{stem}_96h.nc")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        hours, series = catchment_mean_series(path)

        if ref_hours is None:
            ref_hours = hours
        elif not np.allclose(hours, ref_hours):
            raise ValueError(f"{label}: time axis differs from the first forcing")

        conv, _ = file_epoch_and_convention(path, label)
        # Work in interval-END hours throughout, whatever the file declares, so
        # the window masks below have one meaning only.
        h_end = hours + STEP_H if conv == "interval_start" else hours
        if ref_hours_end is None:
            ref_hours_end = h_end
        elif not np.allclose(h_end, ref_hours_end):
            raise SystemExit(f"{label}: interval-end hours differ from the first "
                             "forcing; the inputs mix stamping conventions")

        total = series.sum()
        peak6 = series.max()
        peak_h = float(h_end[int(series.argmax())])
        crit = series[(h_end > CRIT_START_H) & (h_end <= CRIT_END_H)].sum()
        crit_frac = 100.0 * crit / total if total > 0 else np.nan
        burst = series[(h_end > BURST_START_H) & (h_end <= BURST_END_H)].sum()
        burst_frac = 100.0 * burst / total if total > 0 else np.nan

        hyeto[label] = series
        summary.append(
            {
                "Forcing": label,
                "EventTotal_mm": round(total, 1),
                "Peak6h_mm": round(peak6, 1),
                "PeakTime_h": peak_h,
                "Crit24_48h_mm": round(crit, 1),
                "Crit24_48h_pct": round(crit_frac, 1),
                "Burst36_48h_mm": round(burst, 1),
                "Burst36_48h_pct": round(burst_frac, 0),
            }
        )
        print(
            f"{label:11s} total={total:6.1f} mm   peak 6-h={peak6:5.1f} mm "
            f"at +{peak_h:3.0f} h   24-48 h={crit:5.1f} mm ({crit_frac:4.1f} %)"
            f"   burst 36-48 h={burst:5.1f} mm ({burst_frac:3.0f} %)"
        )

    # 1. hyetograph, idx column matching the *_merged.csv convention.
    #    The exported timestamps are EXPLICIT interval bounds. Earlier versions
    #    exported the file's raw stamps while the summary statistics above worked in
    #    interval-END hours, so the CSV and the summaries could disagree by one step
    #    whenever the input convention changed. Both bounds are now written, and
    #    `datetime` is defined as the interval END to match the summary columns and
    #    the way rainfall is conventionally labelled.
    hyd = pd.DataFrame(hyeto)
    hyd.insert(0, "idx", np.arange(len(ref_hours)))
    hyd.insert(1, "hours_end", ref_hours_end)
    hyd.insert(2, "interval_start",
               [START + pd.Timedelta(hours=float(h) - STEP_H) for h in ref_hours_end])
    hyd.insert(3, "datetime",
               [START + pd.Timedelta(hours=float(h)) for h in ref_hours_end])
    hpath = os.path.join(odir, "hyetograph.csv")
    hyd.to_csv(hpath, index=False)
    print("\nWROTE", hpath, hyd.shape)

    # 2. Table 1 summary columns
    smry = pd.DataFrame(summary)
    ref = smry.loc[smry["Forcing"] == "WRF_1p3km", "EventTotal_mm"]
    if not ref.empty and ref.iloc[0] > 0:
        smry["DiffVsWRF_pct"] = (
            100.0 * (smry["EventTotal_mm"] - ref.iloc[0]) / ref.iloc[0]
        ).round(1)
        print(
            f"\nWRF 1.3 km total over the processed window "
            f"({START:%Y-%m-%d %H:%M} to "
            f"{START + pd.Timedelta(hours=float(ref_hours_end[-1]) - STEP_H):%Y-%m-%d %H:%M} UTC): "
            f"{ref.iloc[0]:.1f} mm.\n"
            "This is the rainfall APPLIED to the hydraulic model over the simulated "
            "window, not the calendar-event total; Table 1 must report it as such."
        )
    spath = os.path.join(odir, "forcing_intensity.csv")
    smry.to_csv(spath, index=False)
    print("WROTE", spath)
    print("\n" + smry.to_string(index=False))


if __name__ == "__main__":
    main()
