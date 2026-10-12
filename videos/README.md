# Video supplements: July 2021 Alzette flood

Two synchronized six-forcing animations supporting:

> From weather forecasts to floods: can AI weather models rival physics-based forecasts? A hydraulic evaluation of the July 2021 Luxembourg flood. Haseeb ur Rehman, Félicia Norma Rebecca Teferle and Guy Schumann. NHESS manuscript EGUSPHERE-2026-6277.

- `all_six_3x2_steinsel_discharge_1080p.mp4`: simulated depth maps and observed/simulated discharge at Steinsel.
- `all_six_3x2_walferdange_water_level_1080p.mp4`: simulated depth maps and observed stage/simulated cell depth at Walferdange.
- Smaller `_TIB_1024x576.mp4` copies and poster PNGs are also included.

Download the folder and open `view_comparison.html` to switch videos. Large red markers identify the relevant station in every map. The comparison covers 13 July 2021 18:00 UTC–17 July 2021 18:00 UTC with 17 six-hour snapshots and no temporal interpolation. Playback lasts 27 seconds at 25 fps.

Maps display maximum depth in 40 m blocks derived from 10 m cells, hiding values below 0.10 m. They include river channels and do not validate observed inundation extent. Walferdange observed stage and simulated depth have an uncalibrated datum offset; original values are retained, and curve shape/timing are the relevant diagnostics. AIFS training includes this event.

## Verification and provenance

`NUMERICAL_VERIFICATION.md` and `numerical_audit.json` record 284 passing checks, including byte-identical curve data to the manuscript Figures 4 and 7, raw-cell Walferdange values, all 102 raw depth maps and every 40 m display-block maximum. Steinsel discharge was checked against the existing extraction output; flux integration was not independently repeated. `captions.txt` provides detailed video captions.

The DOI printed inside the videos, https://doi.org/10.5281/zenodo.22749254, identifies the source code/data release v2.0. Release v2.1 adds these videos without changing the numerical results or model configuration.

To rebuild, obtain the raw simulation grids and local research folder described by the manuscript availability statement, install NumPy, Matplotlib and Pillow plus ffmpeg, then run:

```sh
python videos/make_flood_video.py --research-root /path/to/WRF_vs_AI_LISFLOOD_v1 --all-six --station steinsel
python videos/make_flood_video.py --research-root /path/to/WRF_vs_AI_LISFLOOD_v1 --all-six --station walferdange
python videos/audit_video_values.py --research-root /path/to/WRF_vs_AI_LISFLOOD_v1
```

Raw grids, regenerated frame images and display caches are excluded from this repository. Videos, captions and posters are available under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); software scripts retain the repository MIT licence.
