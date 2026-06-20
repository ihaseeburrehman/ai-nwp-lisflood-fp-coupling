#!/usr/bin/env python3
"""
Flood-depth comparison maps for the WRF-vs-AI -> LISFLOOD-FP paper.
3 timestamps (rows) x 5 forcings (cols) of simulated water depth on the 10 m
Alzette domain (coarsened to 50 m for display), with the Alzette river overlaid.
Minimal multi-panel layout: shared depth colourbar, model column titles, time
row labels, one north arrow + scale bar. Style consistent with the paper maps.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.join(HERE, "peak_maps")
RIVER = ("/Users/haseeb.rehman/Documents/Misc/Lisflood_Simulations/"
         "Lisflood_Alzette_river_basin/sub_basins/5m/sub_basin_complete/"
         "pre_processing/alzette_river.shp")
OUT = ("/Users/haseeb.rehman/Documents/Phd_thesis/Research_papers/"
       "WRF_vs_AI_LISFLOOD_v1/figures/figure_flood_maps")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.6,
})

meta = json.load(open(os.path.join(MAPS, "meta.json")))
L, B, R, T = meta["bounds"]
EXTENT = [L, R, B, T]

MODELS = [("wrf", "WRF 1.3 km"), ("wrf12", "WRF 12 km"),
          ("aifs", "AIFS"), ("graphcast", "GraphCast"), ("fuxi", "FuXi")]
TIMES = [(7, "14 Jul 18:00"), (8, "15 Jul 00:00"), (13, "16 Jul 06:00")]
VMIN, VMAX, DRY = 0.1, 2.0, 0.1   # m; cap so floodplain depth gradients show
CELL = meta["cell_m"]

# ---- crop to the active flood corridor (union wet bbox + margin) ----
_all = [np.load(os.path.join(MAPS, f"{m}_{idx:04d}.npy"))
        for idx, _ in TIMES for m, _ in MODELS]
_wet = np.zeros_like(_all[0], dtype=bool)
for a in _all:
    _wet |= np.nan_to_num(a) > DRY
rows = np.where(_wet.any(axis=1))[0]; cols = np.where(_wet.any(axis=0))[0]
mg = 8
r0, r1 = max(rows.min() - mg, 0), min(rows.max() + mg + 1, _wet.shape[0])
c0, c1 = max(cols.min() - mg, 0), min(cols.max() + mg + 1, _wet.shape[1])
# recompute extent for the crop (row0 = north = T)
T2 = T - r0 * CELL; B2 = T - r1 * CELL
L2 = L + c0 * CELL; R2 = L + c1 * CELL
L, R, B, T = L2, R2, B2, T2
EXTENT = [L, R, B, T]
def _crop(a): return a[r0:r1, c0:c1]

# depth colourmap: light->deep blue; dry land grey, outside basin white
depth_cmap = cm.get_cmap("YlGnBu").copy()
depth_cmap.set_bad(alpha=0.0)

# river in EPSG:2169 (raster grid)
try:
    riv = gpd.read_file(RIVER).to_crs(2169)
except Exception as e:
    print("river reproject failed:", e); riv = None

nrow, ncol = len(TIMES), len(MODELS)
fig, axes = plt.subplots(nrow, ncol, figsize=(11.0, 9.4),
                         sharex=True, sharey=True)

for r, (idx, tlab) in enumerate(TIMES):
    for c, (m, mlab) in enumerate(MODELS):
        ax = axes[r, c]
        d = _crop(np.load(os.path.join(MAPS, f"{m}_{idx:04d}.npy")))
        inside = ~np.isnan(d)
        # base: dry land inside basin = light grey
        base = np.where(inside, 0.0, np.nan)
        ax.imshow(np.ma.masked_invalid(base), extent=EXTENT, origin="upper",
                  cmap=ListedColormap(["#ececec"]), vmin=0, vmax=1, zorder=1)
        # flooded depth
        wet = np.where(inside & (d > DRY), d, np.nan)
        im = ax.imshow(np.ma.masked_invalid(wet), extent=EXTENT, origin="upper",
                       cmap=depth_cmap, vmin=VMIN, vmax=VMAX, zorder=2,
                       interpolation="nearest")
        if riv is not None:
            riv.plot(ax=ax, color="#08306b", linewidth=0.35, alpha=0.8, zorder=3)
        ax.set_xlim(L, R); ax.set_ylim(B, T)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#888888")
        if r == 0:
            ax.set_title(mlab, fontsize=10, fontweight="bold", pad=5)
        if c == 0:
            ax.set_ylabel(tlab, fontsize=10, fontweight="bold", labelpad=6)

# north arrow + scale bar on the top-left panel
ax0 = axes[0, 0]
ax0.annotate("", xy=(0.12, 0.93), xytext=(0.12, 0.82), xycoords="axes fraction",
             arrowprops=dict(facecolor="black", width=1.6, headwidth=6))
ax0.text(0.12, 0.945, "N", transform=ax0.transAxes, ha="center", fontsize=8, fontweight="bold")
sb = 10000.0  # 10 km
x0 = L + 0.08 * (R - L); y0 = B + 0.06 * (T - B)
ax0.plot([x0, x0 + sb], [y0, y0], color="black", lw=2.2, solid_capstyle="butt", zorder=5)
ax0.text(x0 + sb / 2, y0 + 0.02 * (T - B), "10 km", ha="center", va="bottom", fontsize=7)

fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.10, wspace=0.06, hspace=0.06)
cax = fig.add_axes([0.30, 0.055, 0.42, 0.018])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
cb.set_label("Simulated water depth (m)", fontsize=9)
cb.ax.tick_params(labelsize=8)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("SAVED", OUT + ".png/.pdf")
