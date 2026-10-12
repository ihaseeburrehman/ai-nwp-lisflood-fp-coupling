# Numerical verification of the two final videos

All 284 checks passed.

- Curve data are byte-identical to the CSVs used by manuscript Figures 4 and 7.
- All 102 water-depth maps (6 forcings × 17 times) were independently re-read and every 40 m display block matched the source maximum exactly.
- Every Walferdange simulated value matches the original 10 m gauge cell, within CSV rounding (0.00005 m).
- Steinsel discharge matches the manuscript extraction output within 0.00005 m³/s; flux integration was not independently repeated.
- All timestamps match the corrected 13 July 2021 18:00 UTC start and six-hour interval.
- All four MP4s decode without errors.

## Peaks shown in the curves

| Forcing | Steinsel peak (m³/s) | Peak hour | Walferdange peak depth (m) | Peak hour |
|---|---:|---:|---:|---:|
| wrf1p3_afterda | 102.5408 | 30 | 2.8800 | 30 |
| wrf12_afterda | 102.7309 | 30 | 2.8820 | 30 |
| wrf12_beforeda | 104.0874 | 30 | 2.8960 | 30 |
| graphcast | 91.2305 | 30 | 2.7960 | 30 |
| aifs | 91.7972 | 30 | 2.7820 | 30 |
| fuxi | 13.7840 | 6 | 1.2350 | 0 |

Observed peaks in manuscript plotting data: Steinsel 131.5000 m³/s; Walferdange stage 3.5120 m.

## Interpretation limits

Walferdange observed stage and simulated depth have an uncalibrated datum offset. The video retains the original values; it does not align their datum artificially. The maps display maximum depth within 40 m blocks and hide depths below 0.10 m; this is a display transformation, not the original 10 m inundation footprint. AIFS training includes this event. These checks confirm consistency with the paper, not independent validation of the hydraulic model or flood extent.
