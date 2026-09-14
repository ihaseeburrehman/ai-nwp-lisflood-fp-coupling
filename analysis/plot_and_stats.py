#!/usr/bin/env python3
"""
plot_and_stats.py  --  discharge hydrographs + performance metrics
WRF-vs-AI → LISFLOOD-FP paper.

Layout
------
* 2×2 panel of discharge hydrographs (Steinsel, Pfaffenthal, Livange,
  Hesperange) with ONE shared legend placed below the panel.
* Separate single panel for Walferdange water-level.
* Colors: dark/light green for WRF (1.3 km / 12 km), orange/purple/red
  for the three AI models.

Run from the paper directory:
    cd /Users/haseeb.rehman/Documents/Phd_thesis/Research_papers/WRF_vs_AI_LISFLOOD_v1
    /opt/homebrew/Caskroom/miniconda/base/bin/python3 plot_and_stats.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr

# ── publication style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":       9,
    "axes.labelsize":  10,
    "axes.titlesize":  10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.linewidth":  0.8,
    "grid.alpha":      0.35,
    "grid.linestyle":  "--",
    "grid.linewidth":  0.6,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR  = os.path.dirname(SCRIPT_DIR)   # <root>/analysis (parent of scripts/)
DATA_DIR   = os.path.join(PAPER_DIR, "data")
os.chdir(PAPER_DIR)
# FIGDIR is defined below, once PAPER_DIR is known; nothing is written here.

# ── model appearance ──────────────────────────────────────────────────────────
# Six rainfall forcings routed through LISFLOOD-FP, plus two operational EFAS
# discharge benchmarks (re-extracted from the source forecast NetCDFs). EFAS is discharge
# only, so it appears at the four discharge gauges, not at the Walferdange level gauge.
MODELS = ["wrf", "wrf12", "wrf12_before", "graphcast", "fuxi", "aifs",
          "efas_ecmwf", "efas_dwd"]
EFAS_MODELS = ["efas_ecmwf", "efas_dwd"]

# All seven rainfall forcings now come from the 96 h rerun set: identical window
# (13-17 Jul 2021), identical start grid and identical LISFLOOD-FP configuration.
# Extractions are written by
# data/lisflood_96h/scripts/extract_station_discharge_96h_all.py.
LF96 = os.path.join(DATA_DIR, "lisflood_96h")
SRC_96H = {
    "wrf":          "wrf1p3_afterda",
    "wrf12":        "wrf12_afterda",
    "wrf12_before": "wrf12_beforeda",
    "graphcast":    "graphcast",
    "fuxi":         "fuxi",
    "aifs":         "aifs",
    "ecmwf":        "ecmwf",
}
MODEL_LABELS = {
    "wrf":        "WRF 1.3 km (After-DA)",
    "wrf12":      "WRF 12 km (After-DA)",
    "wrf12_before": "WRF 12 km (Before-DA)",
    "ecmwf":      "ECMWF HRES",
    "graphcast":  "GraphCast",
    "fuxi":       "FuXi",
    "aifs":       "AIFS",
    "efas_ecmwf": "EFAS (ECMWF)",
    "efas_dwd":   "EFAS (DWD)",
}
MODEL_COLORS = {
    "wrf":        "#1b4332",   # deep forest green – WRF 1.3 km
    "wrf12":      "#52b788",   # lighter mint green – WRF 12 km
    "wrf12_before": "#95d5b2", # palest green – WRF 12 km cold start
    "ecmwf":      "#d62828",   # red – ECMWF operational HRES
    "graphcast":  "#f77f00",   # amber/orange – GraphCast
    "fuxi":       "#7209b7",   # royal purple – FuXi
    "aifs":       "#0077b6",   # sky blue – AIFS
    "efas_ecmwf": "#909090",   # medium grey – operational benchmark (pair)
    "efas_dwd":   "#404040",   # dark grey   – operational benchmark (pair)
}
# pgfplots-friendly RGB (0-255) for the same colours
MODEL_RGB = {
    "wrf":        (27, 67, 50),   "wrf12":      (82, 183, 136),
    "graphcast":  (247, 127, 0),  "fuxi":       (114, 9, 183),
    "wrf12_before": (149, 213, 178), "ecmwf": (214, 40, 40),
    "aifs":       (0, 119, 182),  "efas_ecmwf": (144, 144, 144),
    "efas_dwd":   (64, 64, 64),
}
OBS_RGB = (255, 0, 0)   # matches \definecolor{clrObs}{HTML}{FF0000} in the manuscript
OBS_COLOR = "#FF0000"   # red, identical to the LaTeX hydrograph colour
MODEL_LS = {
    "wrf":        "-",
    "wrf12":      "-",
    "wrf12_before": (0, (5, 2)),
    "ecmwf":      (0, (2, 1, 1, 1)),
    "graphcast":  "-",
    "fuxi":       (0, (4, 1.5)),
    "aifs":       (0, (1, 1)),
    "efas_ecmwf": (0, (3, 1, 1, 1)),
    "efas_dwd":   (0, (3, 1, 1, 1)),
}
MODEL_LW = {m: 1.7 for m in MODELS}
MODEL_LW["wrf"] = 2.2
MODEL_LW["wrf12"] = 1.8
MODEL_MARKER = {m: "o" for m in MODELS}
MODEL_MS     = {m: 3.5 for m in MODELS}

# ── stations ──────────────────────────────────────────────────────────────────
STATIONS = {
    "Walferdange": {
        "type":    "water_level",
        "obs_col": "Relativer Wert [cm]",
        "sim_col": "Walferdange_Depth_cell",
        "unit":    "m",
    },
    "Steinsel": {
        "type":    "discharge",
        "obs_col": "Durchfluss [m³/s]",
        "sim_col": "Steinsel_Q",
        "unit":    "m³/s",
    },
    "Pfaffenthal": {
        "type":    "discharge",
        "obs_col": "Durchfluss [m³/s]",
        "sim_col": "Pfaffenthal_Q",
        "unit":    "m³/s",
    },
    "Livange": {
        "type":    "discharge",
        "obs_col": "Aggregiertes Mittel [m³/s]",
        "sim_col": "Livange_Q",
        "unit":    "m³/s",
    },
    "Hesperange": {
        "type":    "discharge",
        "obs_col": "Aggregiertes Mittel [m³/s]",
        "sim_col": "Hesperange_Q",
        "unit":    "m³/s",
    },
}

OBS_FILES = {
    "Walferdange": (
        "/Volumes/SanDisk 2TB/2026-07-14-115148.previous/Data/Users/haseeb.rehman/Documents/Misc/Data_Datasets/"
        "Stations_and_Observations/Discharge_data_walferdange_2021/"
        "Data_from _AGE/Walferdange.W15.07.2021.csv"
    ),
    "Steinsel": (
        "/Volumes/SanDisk 2TB/2026-07-14-115148.previous/Data/Users/haseeb.rehman/Documents/Misc/Data_Datasets/"
        "Stations_and_Observations/Discharge_data_walferdange_2021/"
        "Data_from _AGE/Steinsel.Q15.VO.07.2021.csv"
    ),
    "Pfaffenthal": (
        "/Volumes/SanDisk 2TB/2026-07-14-115148.previous/Data/Users/haseeb.rehman/Documents/Misc/Data_Datasets/"
        "Stations_and_Observations/Discharge_data_walferdange_2021/"
        "Data_from _AGE/Pfaffenthal.Q15.VO.07.2021.csv"
    ),
    "Livange": (
        "/Volumes/SanDisk 2TB/2026-07-14-115148.previous/Data/Users/haseeb.rehman/Documents/Misc/Data_Datasets/"
        "Stations_and_Observations/Discharge_data_walferdange_2021/"
        "Data_from _AGE/Livange.Q60_07.2021.csv"
    ),
    "Hesperange": (
        "/Volumes/SanDisk 2TB/2026-07-14-115148.previous/Data/Users/haseeb.rehman/Documents/Misc/Data_Datasets/"
        "Stations_and_Observations/Discharge_data_walferdange_2021/"
        "Data_from _AGE/Hesperange.Q60_07.2021.csv"
    ),
}

# Option B window. The hydraulic start grid (13_july_18hr.wd) is the model state
# at 13 July 18:00 UTC, so every run begins there and the 96 h window ends on
# 17 July 18:00. Observations are trimmed to the same window.
START = pd.Timestamp("2021-07-13 18:00")
END   = pd.Timestamp("2021-07-17 18:00")   # 96 h common flood window (idx 0..16)

DISCHARGE_STATIONS = ["Steinsel", "Pfaffenthal", "Livange", "Hesperange"]
GRID = [START + pd.Timedelta(hours=6 * i) for i in range(17)]   # idx 0..16, 6-hourly


# ── helpers ───────────────────────────────────────────────────────────────────
def load_observed(name):
    df = pd.read_csv(
        OBS_FILES[name], sep=";", skiprows=15, decimal=",",
        na_values="---", encoding="latin-1",
    )
    df["Datetime"] = pd.to_datetime(
        df["Datum"] + " " + df["Uhrzeit"], format="%d.%m.%y %H:%M:%S"
    )
    df = df[(df["Datetime"] >= START) & (df["Datetime"] <= END)]
    return df.set_index("Datetime")


# Operational EFAS discharge benchmarks, re-extracted over the full 96 h window
# by scripts/extract_efas_discharge.py from the source forecast NetCDFs. The
# companion paper's CSVs stopped at 16 July 00:00 (13 of the 17 steps).
EFAS_CSV = os.path.join(LF96, "efas", "efas_station_Q.csv")
EFAS_COL = {"efas_ecmwf": "EFAS_ECMWF", "efas_dwd": "EFAS_DWD"}


def load_efas(model):
    """Build an EFAS discharge series in our format.
    EFAS is discharge only, so only the four discharge gauges are populated."""
    if not os.path.exists(EFAS_CSV):
        raise SystemExit(f"{EFAS_CSV} missing; run scripts/extract_efas_discharge.py")
    key = EFAS_COL[model]
    d = pd.read_csv(EFAS_CSV)
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d.set_index("datetime")
    frames = {f"{st}_Q": d[f"{st}_{key}"]
              for st in ["Steinsel", "Pfaffenthal", "Livange", "Hesperange"]}
    return pd.DataFrame(frames)


def load_simulated(model):
    if model in EFAS_MODELS:
        return load_efas(model)
    # Surveyed per-gauge river widths (Sect. 3.5): the discharge transect is the
    # width measured at each gauge from winter orthoimagery, not a single 20 m
    # constant. Written by scripts/extract_discharge_surveyed_widths.py.
    fn = os.path.join(LF96, "line_integral_18utc", f"{SRC_96H[model]}_station_Q.csv")
    if not os.path.exists(fn):
        return None
    df = pd.read_csv(fn)
    df["Time"] = pd.to_datetime(df["Time"])
    return df.set_index("Time")


def nse(obs, sim):
    denom = np.sum((obs - obs.mean()) ** 2)
    return 1 - np.sum((obs - sim) ** 2) / denom if denom else np.nan


def kge(obs, sim):
    """Modified Kling-Gupta efficiency, KGE' (Kling et al., 2012).

    The variability term is the ratio of coefficients of variation, not of standard
    deviations as in the original Gupta et al. (2009) formulation, so the bias and
    variability terms are not cross-correlated. This matters here because several
    forcings carry a large volume deficit. The manuscript states the same definition.
    """
    if len(obs) < 2 or obs.std() == 0 or sim.mean() == 0 or obs.mean() == 0:
        return np.nan
    r, _ = pearsonr(obs, sim)
    beta  = sim.mean() / obs.mean()
    gamma = (sim.std() / sim.mean()) / (obs.std() / obs.mean())
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)


def rmse(obs, sim):
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def mae(obs, sim):
    return float(np.mean(np.abs(sim - obs)))


def smape(obs, sim):
    # Symmetric MAPE (%). Bounded 0-200%; robust at low baseflow where plain
    # MAPE diverges (obs -> 0). Matches the companion paper's error metric.
    denom = (np.abs(sim) + np.abs(obs)) / 2.0
    mask = denom > 1e-9
    if not mask.any():
        return np.nan
    return float(100.0 * np.mean(np.abs(sim[mask] - obs[mask]) / denom[mask]))


def bias(obs, sim):
    # Mean signed error (sim - obs); same units as the variable.
    return float(np.mean(sim - obs))


def fmt_ax(ax, title=""):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.set_xlim(START, END)
    ax.grid(True)
    if title:
        ax.set_title(title, fontweight="bold", pad=4)
    ax.tick_params(axis="x", labelsize=7.5)


# ── load all data ─────────────────────────────────────────────────────────────
obs_data   = {s: load_observed(s) for s in STATIONS}
model_data = {m: load_simulated(m) for m in MODELS}


def validate_inputs():
    """Check every required input BEFORE any metric is computed.

    Three failures used to survive as far as the printed table: a missing run came
    through as a column of NaNs, a run extracted under a different epoch produced
    plausible numbers from a partial overlap, and metrics were printed before the
    per-series guard was ever reached. Everything is therefore checked up front and
    the script aborts rather than reporting anything it cannot stand behind.
    """
    problems = []

    # 1. the extraction manifest must agree with the window configured here
    man = os.path.join(LF96, "line_integral_18utc", "run_manifest.json")
    if os.path.exists(man):
        with open(man) as fh:
            m = json.load(fh)
        got = pd.Timestamp(m.get("assumed_start_utc", "").replace("Z", ""))
        if got != START:
            problems.append(
                f"run_manifest.json says the CSVs were extracted for "
                f"{got:%Y-%m-%d %H:%M}, but this script is configured for "
                f"{START:%Y-%m-%d %H:%M}")
        unver = [r["model"] for r in m.get("runs", []) if not r.get("epoch_verified")]
        if unver:
            print(f"  note: epoch not verified against the forcing for: "
                  f"{', '.join(unver)}")
    else:
        problems.append(f"no run_manifest.json in {os.path.dirname(man)}; "
                        "re-run extract_discharge_line_integral.py")

    # 2. every model must be present and cover the whole window
    for mdl in MODELS:
        df = model_data.get(mdl)
        if df is None or len(df) == 0:
            problems.append(f"{mdl}: no simulated series loaded")
            continue
        for st in DISCHARGE_STATIONS:
            col = f"{st}_Q"
            if col not in df.columns:
                problems.append(f"{mdl}: column {col} missing")
                continue
            miss = df[col].reindex(GRID).isna()
            if miss.any():
                first = [f"{t:%Y-%m-%d %H:%M}" for t, v in miss.items() if v][0]
                problems.append(
                    f"{mdl}/{st}: {int(miss.sum())} of {len(GRID)} steps missing "
                    f"(first {first})")

    if problems:
        raise SystemExit("Refusing to compute metrics:\n  - " +
                         "\n  - ".join(problems))
    print(f"input validation passed: {len(MODELS)} forcings x {len(GRID)} steps, "
          f"{START:%Y-%m-%d %H:%M} to {END:%Y-%m-%d %H:%M} UTC")


validate_inputs()

# ── performance metrics ───────────────────────────────────────────────────────
metrics = {}
print("\nPerformance metrics\n" + "=" * 60)
for station, info in STATIONS.items():
    obs_df  = obs_data[station]
    obs_col = info["obs_col"]
    is_wl   = info["type"] == "water_level"
    obs_ts  = obs_df[obs_col].dropna()

    obs_peak_raw  = obs_ts.max()
    obs_peak_time = obs_ts.idxmax()
    print(f"\n{station}  (obs peak = {obs_peak_raw:.1f} "
          f"{'cm' if is_wl else 'm³/s'} @ {obs_peak_time})")

    for model in MODELS:
        df = model_data[model]
        if df is None:
            continue
        sim_col = info["sim_col"]
        if sim_col not in df.columns:      # e.g. EFAS has no Walferdange level
            continue
        aligned = pd.concat(
            [obs_df[obs_col], df[sim_col]], axis=1, join="inner"
        ).dropna()
        if aligned.empty:
            continue
        o = aligned.iloc[:, 0].values.astype(float)
        s = aligned.iloc[:, 1].values.astype(float)

        if is_wl:
            o_cmp = o / 100.0
            s_cmp = s
            obs_pk = obs_peak_raw / 100.0
            sim_pk = df[sim_col].max()
        else:
            o_cmp  = o
            s_cmp  = s
            obs_pk = obs_peak_raw
            sim_pk = df[sim_col].max()

        nse_v = nse(o_cmp, s_cmp)
        kge_v = kge(o_cmp, s_cmp)
        pde   = (sim_pk - obs_pk) / obs_pk * 100.0
        t_err = (df[sim_col].idxmax() - obs_peak_time) / pd.Timedelta(hours=1)

        metrics[(station, model)] = dict(
            nse=nse_v, kge=kge_v, pde=pde, timing=t_err, sim_peak=sim_pk,
            rmse=rmse(o_cmp, s_cmp), mae=mae(o_cmp, s_cmp),
            smape=smape(o_cmp, s_cmp), bias=bias(o_cmp, s_cmp),
        )
        flag = "*" if df is not None and len(df) < 17 else ""  # incomplete sim (<96 h)
        print(f"  {MODEL_LABELS[model]:30s}  NSE={nse_v:+.2f}  KGE={kge_v:+.2f}"
              f"  PDE={pde:+.1f}%  Δt={t_err:+.1f}h{flag}")

print("\n* = simulation did not cover full 96 h window (metrics based on available steps)")


# ══════════════════════════════════════════════════════════════════════════════
#  Emit the data files the manuscript's inline pgfplots read:
#    data/pgfplots/<station>_merged.csv   (6-hourly discharge, idx 0..12)
#    data/pgfplots/walferdange_merged.csv (6-hourly water level / depth, m)
#    data/pgfplots/averaged_metrics.csv   (per-forcing skill + error metrics)
#  plus the Taylor diagram PDF (figures/taylor_diagram.pdf).  The .tex is untouched.
# ══════════════════════════════════════════════════════════════════════════════
PGFDIR = os.path.join(DATA_DIR, "pgfplots")
# figures live with the manuscript: <root>/manuscript/figures
FIGDIR = os.path.join(PAPER_DIR, os.pardir, "manuscript", "figures")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(PGFDIR, exist_ok=True)
MERGED_COL = {"wrf": "WRF_1p3km", "wrf12": "WRF_12km", "wrf12_before": "WRF_12km_BeforeDA",
              "graphcast": "GraphCast",
              "fuxi": "FuXi", "aifs": "AIFS", "efas_ecmwf": "EFAS_ECMWF", "efas_dwd": "EFAS_DWD"}
CSVNAME = {"wrf": "wrf", "wrf12": "wrf12", "wrf12_before": "wrf12before",
           "graphcast": "graphcast", "fuxi": "fuxi",
           "aifs": "aifs", "efas_ecmwf": "efasecmwf", "efas_dwd": "efasdwd"}


def _obs_grid(station, scale=1.0):
    obs = obs_data[station][STATIONS[station]["obs_col"]].dropna().sort_index()
    s = obs.reindex(GRID, method="nearest", tolerance=pd.Timedelta("1h"))
    return (s * scale).values


def _model_grid(model, station, suffix="_Q"):
    df = model_data[model]
    if df is None or f"{station}{suffix}" not in df.columns:
        return np.full(17, np.nan)
    col = df[f"{station}{suffix}"].reindex(GRID)
    if col.isna().any():
        missing = [f"{t:%Y-%m-%d %H:%M}" for t, v in col.items() if pd.isna(v)]
        raise SystemExit(
            f"{station}{suffix}: the simulated series does not cover the whole "
            f"analysis window {GRID[0]:%Y-%m-%d %H:%M} to {GRID[-1]:%Y-%m-%d %H:%M}; "
            f"missing {len(missing)} step(s), first {missing[0]}.\n"
            "PARTIAL overlap is the dangerous case: it yields plausible-looking but "
            "wrong metrics. The extracted CSVs were produced under a different run "
            "epoch than the one configured here. Re-run "
            "extract_discharge_line_integral.py against runs that start at "
            f"{GRID[0]:%Y-%m-%d %H:%M} before analysing.")
    return col.values


def _fmt(v):
    return "nan" if not np.isfinite(v) else f"{v:.4f}"


# ── discharge merged CSVs (4 gauges) ─────────────────────────────────────────
for st in DISCHARGE_STATIONS:
    cols = ["idx", "Observed"] + [MERGED_COL[m] for m in MODELS]
    obs = _obs_grid(st)
    series = {MERGED_COL[m]: _model_grid(m, st, "_Q") for m in MODELS}
    lines = [",".join(cols)]
    for i in range(17):
        row = [str(i), _fmt(obs[i])] + [_fmt(series[MERGED_COL[m]][i]) for m in MODELS]
        lines.append(",".join(row))
    open(os.path.join(PGFDIR, f"{st.lower()}_merged.csv"), "w").write("\n".join(lines) + "\n")
print("Wrote 4 discharge *_merged.csv")

# ── Walferdange water level (m); EFAS omitted (discharge only) ────────────────
WL = [("wrf", "WRF_1p3km_m"), ("wrf12", "WRF_12km_m"),
      ("wrf12_before", "WRF_12km_BeforeDA_m"), ("graphcast", "GraphCast_m"),
      ("fuxi", "FuXi_m"), ("aifs", "AIFS_m")]
obs = _obs_grid("Walferdange", scale=0.01)
wl_series = {c: _model_grid(m, "Walferdange", "_Depth_cell") for m, c in WL}
lines = ["idx,Observed_m," + ",".join(c for _, c in WL)]
for i in range(17):
    row = [str(i), _fmt(obs[i])] + [_fmt(wl_series[c][i]) for _, c in WL]
    lines.append(",".join(row))
open(os.path.join(PGFDIR, "walferdange_merged.csv"), "w").write("\n".join(lines) + "\n")
print("Wrote walferdange_merged.csv")


# ── per-station metrics (the values quoted station-by-station in the text) ───
KEYS_ST = ["nse", "kge", "pde", "timing", "rmse", "mae", "smape", "bias"]
_st_lines = ["station,model," + ",".join(KEYS_ST)]
for _s in DISCHARGE_STATIONS:
    for _m in MODELS:
        if (_s, _m) not in metrics:
            continue
        _r = metrics[(_s, _m)]
        _st_lines.append(f"{_s},{CSVNAME[_m]}," +
                         ",".join(_fmt(_r.get(k, float("nan"))) for k in KEYS_ST))
open(os.path.join(PGFDIR, "metrics_by_station.csv"), "w").write("\n".join(_st_lines) + "\n")
print("Wrote metrics_by_station.csv")

# ── averaged metrics across the four discharge gauges ────────────────────────
def agg(key):
    out = {}
    for m in MODELS:
        vals = [metrics[(s, m)][key] for s in DISCHARGE_STATIONS
                if (s, m) in metrics and np.isfinite(metrics[(s, m)].get(key, np.nan))]
        out[m] = float(np.mean(vals)) if vals else np.nan
    return out


KEYS = ["nse", "kge", "pde", "timing", "rmse", "mae", "smape", "bias"]
A = {k: agg(k) for k in KEYS}
lines = ["model," + ",".join(KEYS)]
for m in MODELS:
    lines.append(CSVNAME[m] + "," + ",".join(_fmt(A[k][m]) for k in KEYS))
open(os.path.join(PGFDIR, "averaged_metrics.csv"), "w").write("\n".join(lines) + "\n")
print("Wrote averaged_metrics.csv")
print("\nAveraged metrics (4 discharge gauges):")
for m in MODELS:
    print(f"  {MODEL_LABELS[m]:24s} NSE={A['nse'][m]:+.2f} KGE={A['kge'][m]:+.2f} "
          f"PDE={A['pde'][m]:+.1f}% PTE={A['timing'][m]:+.1f}h "
          f"RMSE={A['rmse'][m]:.1f} MAE={A['mae'][m]:.1f} SMAPE={A['smape'][m]:.0f}% bias={A['bias'][m]:+.1f}")


# ── Taylor diagram (matplotlib PDF) ──────────────────────────────────────────
def emit_taylor():
    pts = {}
    for m in MODELS:
        df = model_data[m]
        if df is None:
            continue
        O, S = [], []
        for st in DISCHARGE_STATIONS:
            if f"{st}_Q" not in df.columns:
                continue
            al = pd.concat([obs_data[st][STATIONS[st]["obs_col"]], df[f"{st}_Q"]],
                           axis=1, join="inner").dropna()
            if len(al) < 3:
                continue
            oo, ss = al.iloc[:, 0].values.astype(float), al.iloc[:, 1].values.astype(float)
            if oo.std() > 0:
                O.append(oo / oo.std())
                S.append(ss / oo.std())
        if not O:
            continue
        O, S = np.concatenate(O), np.concatenate(S)
        pts[m] = (max(float(np.corrcoef(O, S)[0, 1]), -0.999), float(S.std() / O.std()))

    # ── Cartesian Taylor diagram (Taylor, 2001) ──────────────────────────────
    # A point at correlation R and normalised std sigma plots at
    #   x = sigma*R,  y = sigma*sqrt(1-R^2);  the observed reference is at (1, 0).
    rmax = max(1.5, max((s for _, s in pts.values()), default=1.4) * 1.12)
    fig, ax = plt.subplots(figsize=(7.0, 6.4))

    arc = np.linspace(0, np.pi / 2, 200)
    # standard-deviation arcs (centred on origin)
    for sd in np.arange(0.25, rmax + 1e-9, 0.25):
        ax.plot(sd * np.cos(arc), sd * np.sin(arc), color="0.85", lw=0.6, zorder=1)
    # reference standard-deviation arc (sigma = 1)
    ax.plot(np.cos(arc), np.sin(arc), color="0.45", ls="--", lw=1.3, zorder=2)

    # correlation spokes + outer labels
    cors = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    for c in cors:
        th = np.arccos(c)
        ax.plot([0, rmax * np.cos(th)], [0, rmax * np.sin(th)],
                color="0.88", lw=0.6, zorder=1)
        ax.text((rmax + 0.045) * np.cos(th), (rmax + 0.045) * np.sin(th), f"{c:g}",
                ha="center", va="center", fontsize=12.0, color="0.30")
    ax.text((rmax + 0.20) * np.cos(np.pi / 4), (rmax + 0.20) * np.sin(np.pi / 4),
            "Correlation", rotation=-45, ha="center", va="center", fontsize=12.0)

    # centred-RMSD skill contours (circles centred on the reference point)
    for rms in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        x = 1 + rms * np.cos(arc * 2)        # full semicircle
        y = rms * np.sin(arc * 2)
        keep = (x >= 0) & (np.hypot(x, y) <= rmax)
        if keep.any():
            ax.plot(x[keep], y[keep], color="#2a8f3c", ls=":", lw=0.9, alpha=0.55, zorder=1)

    # observed reference and model markers
    ax.plot(1.0, 0.0, marker="*", ms=19, color="#FF0000", mec="black", mew=0.7,
            zorder=6, label="Observed", ls="None")
    MARK = {"wrf": "o", "wrf12": "s", "graphcast": "^", "fuxi": "D",
            "aifs": "v", "efas_ecmwf": "X", "efas_dwd": "P"}
    for m, (R, s) in pts.items():
        ax.plot(s * R, s * np.sqrt(max(1 - R ** 2, 0.0)),
                marker=MARK.get(m, "o"), ms=11, color=MODEL_COLORS[m],
                mec="black", mew=0.7, ls="None", zorder=5, label=MODEL_LABELS[m])

    ax.set_xlim(0, rmax + 0.06)
    ax.set_ylim(0, rmax + 0.06)
    ax.set_aspect("equal")
    ax.set_xlabel("Normalised standard deviation", fontsize=12.0)
    ax.set_ylabel("Normalised standard deviation", fontsize=12.0)
    ax.tick_params(axis="both", labelsize=12.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=12.0,
              framealpha=0.95, edgecolor="0.7", labelspacing=1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "taylor_diagram.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "taylor_diagram.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Wrote figures/taylor_diagram.pdf  (.png)")


emit_taylor()
