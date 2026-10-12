#!/usr/bin/env python3
"""Independent numerical audit of animation inputs against manuscript and raw grids."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import csv,json,hashlib,subprocess,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--research-root',type=Path,default=ROOT);ROOT=parser.parse_args().research_root.resolve()
LF=ROOT/'analysis/data/lisflood_96h';PAPER=ROOT/'manuscript_NHESS'
models={'wrf1p3_afterda':'WRF_1p3km','wrf12_afterda':'WRF_12km','wrf12_beforeda':'WRF_12km_BeforeDA','graphcast':'GraphCast','aifs':'AIFS','fuxi':'FuXi'}
checks=[];peaks=[]
def check(ok,message):
 checks.append({'passed':bool(ok),'check':message})
 if not ok:raise AssertionError(message)
def table(p):return np.genfromtxt(p,delimiter=',',names=True)
for station in ['steinsel','walferdange']:
 p=ROOT/'analysis/data/pgfplots'/f'{station}_merged.csv';q=PAPER/'plot_data'/p.name
 check(p.read_bytes()==q.read_bytes(),f'{station}: video input is byte-identical to manuscript figure data')
 check(p.name in (PAPER/'figure_sources'/('fig07_waterlevel.tex' if station=='walferdange' else 'fig04_hydrographs.tex')).read_text(),f'{station}: figure source uses the verified CSV')
ste=table(ROOT/'analysis/data/pgfplots/steinsel_merged.csv');wal=table(ROOT/'analysis/data/pgfplots/walferdange_merged.csv')
check(np.array_equal(ste['idx'],np.arange(17)) and np.array_equal(wal['idx'],np.arange(17)),'Both plotted time indices cover 0..16 without missing or reordered rows')
manifest=json.loads((LF/'line_integral_18utc/run_manifest.json').read_text());start=datetime.fromisoformat(manifest['assumed_start_utc'])
check(start==datetime(2021,7,13,18,tzinfo=timezone.utc),'Initialization is 13 July 2021 18:00 UTC')
for model,col in models.items():
 record=next(r for r in manifest['runs'] if r['model']==model)
 check(record['epoch_verified'] and record['interval_h']==6 and record['steps']==17 and record['forcing_convention']=='interval_start',f'{model}: corrected 96-hour run, verified epoch and six-hour interval')
 with (LF/'line_integral_18utc'/f'{model}_station_Q.csv').open() as f: rows=list(csv.DictReader(f))
 check(all(datetime.fromisoformat(r['Time']).replace(tzinfo=timezone.utc)==start+timedelta(hours=6*i) for i,r in enumerate(rows)),f'{model}: extracted timestamps match the video')
 check(np.allclose(ste[col],[float(r['Steinsel_Q']) for r in rows],rtol=0,atol=.00005),f'{model}: all Steinsel curve values match extracted discharge within CSV precision')
 check(np.allclose(wal[col+'_m'],[float(r['Walferdange_Depth_cell']) for r in rows],rtol=0,atol=.00005),f'{model}: all Walferdange curve values match extracted cell depth within CSV precision')
 depths=[]
 for step in range(17):
  p=LF/'runs'/model/'results'/f'6hr-{step:04d}.wd'
  with p.open() as f:
   header={}
   for _ in range(6):
    k,v=f.readline().split();header[k.lower()]=float(v)
  nr,nc=int(header['nrows']),int(header['ncols'])
  # Independently parse the source using loadtxt rather than the renderer's fromstring.
  raw=np.loadtxt(p,skiprows=6,dtype=np.float32)
  check(raw.shape==(nr,nc),f'{model} step {step}: raw raster dimensions')
  rr=int(np.floor((header['yllcorner']+nr*header['cellsize']-81571)/header['cellsize']))
  cc=int(np.floor((77256-header['xllcorner'])/header['cellsize']))
  depths.append(max(float(raw[rr,cc]),0))
  # Independent 40 m reduction: four row/column phases, not reshape/max axes.
  reduced=np.zeros((nr//4,nc//4),dtype=np.float32)
  for dr in range(4):
   for dc in range(4):
    part=raw[dr:(nr//4)*4:4,dc:(nc//4)*4:4]
    np.maximum(reduced,np.where((part>=0)&(part!=header['nodata_value']),part,0),out=reduced)
  cache=np.load(OUT/'cache'/model/f'6hr-{step:04d}.npy')
  check(np.array_equal(reduced,cache),f'{model} step {step}: every displayed block matches raw-grid maximum exactly')
 check(np.allclose(depths,wal[col+'_m'],rtol=0,atol=.00005),f'{model}: all 17 Walferdange values also match the RAW 10 m gauge cell')
 peaks.append({'forcing':model,'steinsel_peak_m3_s':float(ste[col].max()),'steinsel_peak_elapsed_h':int(ste[col].argmax())*6,'walferdange_peak_depth_m':float(wal[col+'_m'].max()),'walferdange_peak_elapsed_h':int(wal[col+'_m'].argmax())*6})
 print(model,'all 17 raw grids and both curves verified',flush=True)
for station,suffix in [('Steinsel','all_six_3x2_steinsel_discharge'),('Walferdange','all_six_3x2_walferdange_water_level')]:
 meta=json.loads((OUT/(suffix+'_metadata.json')).read_text())
 check(meta['station']==station and len(meta['steps'])==17,f'{station}: frame metadata identifies correct station and 17 frames')
 for i,entry in enumerate(meta['steps']):
  check(entry['valid_utc']==(start+timedelta(hours=6*i)).isoformat().replace('+00:00','Z'),f'{station} frame {i}: timestamp label')
 for size in ['1080p','TIB_1024x576']:
  p=OUT/(suffix+'_'+size+'.mp4')
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,r_frame_rate,nb_frames','-show_entries','format=duration','-of','json',str(p)]))
  stream=probe['streams'][0];expected=(1920,1080) if size=='1080p' else (1024,576)
  check((stream['width'],stream['height'])==expected and stream['r_frame_rate']=='25/1',f'{p.name}: correct resolution and 25 fps')
  subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  check(True,f'{p.name}: complete playback decodes without errors')
report={'passed':all(c['passed'] for c in checks),'checks_count':len(checks),'checks':checks,'peaks':peaks,'observed_peaks':{'steinsel_m3_s':float(ste['Observed'].max()),'walferdange_stage_m':float(wal['Observed_m'].max())},'scope':'Curve equality to manuscript plotting data, timestamp checks, every block in 102 raw depth rasters, raw Walferdange gauge values, and playback. This does not independently repeat discharge flux integration or validate the scientific model against observed flood extent.'}
(OUT/'numerical_audit.json').write_text(json.dumps(report,indent=2))
text=['# Numerical verification of the two final videos','',f"All {len(checks)} checks passed.",'','- Curve data are byte-identical to the CSVs used by manuscript Figures 4 and 7.','- All 102 water-depth maps (6 forcings × 17 times) were independently re-read and every 40 m display block matched the source maximum exactly.','- Every Walferdange simulated value matches the original 10 m gauge cell, within CSV rounding (0.00005 m).','- Steinsel discharge matches the manuscript extraction output within 0.00005 m³/s; flux integration was not independently repeated.','- All timestamps match the corrected 13 July 2021 18:00 UTC start and six-hour interval.','- All four MP4s decode without errors.','','## Peaks shown in the curves','','| Forcing | Steinsel peak (m³/s) | Peak hour | Walferdange peak depth (m) | Peak hour |','|---|---:|---:|---:|---:|']
for r in peaks:text.append(f"| {r['forcing']} | {r['steinsel_peak_m3_s']:.4f} | {r['steinsel_peak_elapsed_h']} | {r['walferdange_peak_depth_m']:.4f} | {r['walferdange_peak_elapsed_h']} |")
text+=['',f"Observed peaks in manuscript plotting data: Steinsel {report['observed_peaks']['steinsel_m3_s']:.4f} m³/s; Walferdange stage {report['observed_peaks']['walferdange_stage_m']:.4f} m.",'','## Interpretation limits','','Walferdange observed stage and simulated depth have an uncalibrated datum offset. The video retains the original values; it does not align their datum artificially. The maps display maximum depth within 40 m blocks and hide depths below 0.10 m; this is a display transformation, not the original 10 m inundation footprint. AIFS training includes this event. These checks confirm consistency with the paper, not independent validation of the hydraulic model or flood extent.']
(OUT/'NUMERICAL_VERIFICATION.md').write_text('\n'.join(text)+'\n')
print('PASS:',len(checks),'checks',flush=True)
