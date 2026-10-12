#!/usr/bin/env python3
"""Reproduce an animation from the corrected, manuscript-linked LISFLOOD-FP runs."""
from pathlib import Path
from datetime import datetime, timedelta
import json, subprocess, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LightSource
from matplotlib.cm import ScalarMappable
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
LF=ROOT/'analysis/data/lisflood_96h'
OUT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--all-six',action='store_true')
parser.add_argument('--research-root',type=Path,default=ROOT,help='Research folder containing analysis/data/lisflood_96h')
parser.add_argument('--station',choices=['steinsel','walferdange'],default='steinsel')
args=parser.parse_args()
ROOT=args.research_root.resolve();LF=ROOT/'analysis/data/lisflood_96h'
WALF=args.station=='walferdange'
STATION='Walferdange' if WALF else 'Steinsel'
OBS='Observed_m' if WALF else 'Observed'
POINT=(77.256,81.571) if WALF else (77.432,82.659)
MODELS=[('wrf1p3_afterda','WRF · 1.3 km','WRF_1p3km','#D55E00'),('graphcast','GraphCast','GraphCast','#009E73'),('aifs','AIFS','AIFS','#0072B2'),('fuxi','FuXi','FuXi','#CC79A7')]
if args.all_six:
 MODELS=[MODELS[0],('wrf12_afterda','WRF · 12 km after DA','WRF_12km','#E69F00'),('wrf12_beforeda','WRF · 12 km before DA','WRF_12km_BeforeDA','#56B4E9')]+MODELS[1:]
SUFFIX=('all_six_3x2' if len(MODELS)==6 else 'primary_2x2')+'_'+args.station+('_water_level' if WALF else '_discharge')
manifest=json.loads((LF/'line_integral_18utc/run_manifest.json').read_text())
start=datetime.fromisoformat(manifest['assumed_start_utc'])
for model,*_ in MODELS:
 r=next(r for r in manifest['runs'] if r['model']==model)
 assert r['epoch_verified'] and r['forcing_convention']=='interval_start'
 assert r['steps']==17 and r['interval_h']==6
p=LF/'runs/graphcast/results/6hr-0000.wd'
with p.open() as f:
 header={}
 for _ in range(6):
  k,v=f.readline().split();header[k.lower()]=float(v)
nr,nc=int(header['nrows']),int(header['ncols']); factor=4
# Max pooling preserves narrow wet channels for display; source grids remain unchanged.
def poolmax(a):
 r,c=a.shape;r-=r%factor;c-=c%factor
 return a[:r,:c].reshape(r//factor,factor,c//factor,factor).max(axis=(1,3))
def read_depth(p):
 cache=OUT/'cache'/p.parent.parent.name/(p.stem+'.npy')
 if cache.exists(): return np.load(cache)
 with p.open('rb') as f:
  h={}
  for _ in range(6):
   k,v=f.readline().split();h[k.decode().lower()]=float(v)
  assert h==header, f'Grid mismatch: {p}'
  a=np.fromstring(f.read(),sep=' ',dtype=np.float32)
 assert a.size==nr*nc
 a=a.reshape(nr,nc);a=np.where((a>=0)&(a!=header['nodata_value']),a,0)
 result=poolmax(a)
 cache.parent.mkdir(parents=True,exist_ok=True);np.save(cache,result)
 return result
bed=np.load(LF/'bed_elevation.npy');mask=np.load(LF/'basin_mask.npy')
assert bed.shape==mask.shape==(nr,nc)
valid=(bed>0)&mask
# Crop to basin envelope, then add a small consistent border.
rr,cc=np.nonzero(mask); r0=max(0,rr.min()//factor-3);r1=min(nr//factor,rr.max()//factor+4)
c0=max(0,cc.min()//factor-3);c1=min(nc//factor,cc.max()//factor+4)
bs=bed[:nr//factor*factor,:nc//factor*factor].reshape(nr//factor,factor,nc//factor,factor).mean(axis=(1,3))
ms=poolmax(mask)[r0:r1,c0:c1]
bg=np.where(ms,bs[r0:r1,c0:c1],np.nan)
x0=header['xllcorner'];top=header['yllcorner']+nr*10
extent=[(x0+c0*40)/1000,(x0+c1*40)/1000,(top-r1*40)/1000,(top-r0*40)/1000]
series=np.genfromtxt(ROOT/'analysis/data/pgfplots'/f'{args.station}_merged.csv',delimiter=',',names=True)
assert len(series)==17
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig=plt.figure(figsize=(19.2,10.8),dpi=100,facecolor='white')
cols=2 if len(MODELS)==4 else 3
gs=fig.add_gridspec(2,cols,left=.035,right=.925,top=.845,bottom=.14,hspace=.24,wspace=.13)
axes=[];ims=[];cursors=[]
cmap=plt.get_cmap('viridis').copy();cmap.set_bad((0,0,0,0));norm=Normalize(.1,3,clip=True)
hours=np.arange(17)*6
for i,(m,label,col,color) in enumerate(MODELS):
 cell=gs[i//cols,i%cols]
 inner=cell.subgridspec(1,2,width_ratios=[1,1.3],wspace=.23)
 ax=fig.add_subplot(inner[0]);axes.append(ax)
 ax.imshow(bg,cmap='Greys',vmin=180,vmax=470,extent=extent,origin='upper',alpha=.35)
 im=ax.imshow(np.full(bg.shape,np.nan),cmap=cmap,norm=norm,extent=extent,origin='upper',interpolation='nearest');ims.append(im)
 ax.contour(ms.astype(float),levels=[.5],colors='#727b83',linewidths=.45,extent=extent,origin='upper')
 ax.scatter([POINT[0]],[POINT[1]],marker='o',s=145,c='#e60000',edgecolors='white',linewidths=2,zorder=6)
 ax.scatter([POINT[0]],[POINT[1]],marker='o',s=175,facecolors='none',edgecolors='#7a0000',linewidths=.8,zorder=7)
 ax.set_title(label,fontsize=16,fontweight='bold',color=color,pad=12)
 ax.set_xlabel('Easting (km)',fontsize=10);ax.set_ylabel('Northing (km)',fontsize=10)
 ax.tick_params(labelsize=9);ax.set_aspect('equal')
 # Each map is paired with its gauge response; all panels use identical axes.
 right=inner[1].subgridspec(3,1,height_ratios=[.35,1,.28])
 axq=fig.add_subplot(right[1]);axq.plot(hours,series[OBS],color='#222222',lw=2,label='Observed stage' if WALF else 'Observed')
 axq.plot(hours,series[col+'_m' if WALF else col],color=color,lw=2,label='Simulated depth' if WALF else 'Simulated')
 axq.set_xlim(0,96);axq.set_ylim((0,4) if WALF else (-5,150))
 axq.set_ylabel('Water level / depth (m)' if WALF else 'Discharge (m³/s)',fontsize=10);axq.set_xlabel('Elapsed time (hours)',fontsize=10);axq.tick_params(labelsize=9);axq.grid(alpha=.18)
 axq.set_title(STATION+' gauge',loc='left',fontsize=12)
 axq.legend(loc='upper right',frameon=False,fontsize=9)
 cursors.append(axq.axvline(0,color='#555555',ls='--',lw=1.3))
 fig.add_subplot(right[0]).axis('off');fig.add_subplot(right[2]).axis('off')
cax=fig.add_axes([.946,.30,.012,.45]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,extend='max');cb.set_label('Simulated water depth (m)',fontsize=12)
fig.text(.035,.957,'AI and physics-based weather forecasts → simulated flood response',fontsize=24,fontweight='bold',color='#172e44')
fig.text(.035,.918,'July 2021 Alzette flood, Luxembourg · '+STATION+(' observed stage and simulated depth' if WALF else ' river discharge')+' · all six forcings' if args.all_six else STATION+' · four primary forcings',fontsize=15,color='#425569')
time=fig.text(.925,.876,'',ha='right',fontsize=14,fontweight='bold',color='#172e44')
fig.text(.035,.074,'10 m grids · 6-hour snapshots · no temporal interpolation · red marker: '+STATION+' · EPSG:2169'+(' · Stage/depth datum offset is uncalibrated.' if WALF else ''),fontsize=11,color='#425569')
fig.text(.035,.047,'Display: maximum depth in 40 m blocks; <0.10 m hidden. Maps include river channels; flood extent is not observation-validated. AIFS training includes this event.',fontsize=10,color='#425569')
fig.text(.035,.020,'Corrected forcing, release v2.0 · Haseeb ur Rehman et al. · DOI: 10.5281/zenodo.22749254 · start: 13 July 2021, 18:00 UTC',fontsize=10,color='#425569')
frames=OUT/('frames_'+SUFFIX);frames.mkdir(exist_ok=True)
stats=[]
for step in range(17):
 entry={'step':step,'valid_utc':(start+timedelta(hours=6*step)).isoformat().replace('+00:00','Z')}
 for i,(m,*_) in enumerate(MODELS):
  a=read_depth(LF/'runs'/m/'results'/f'6hr-{step:04d}.wd')[r0:r1,c0:c1]
  entry[m]={'display_max_depth_m':float(a[ms].max())}
  ims[i].set_data(np.ma.masked_where((a<.1)|~ms,a))
 time.set_text((start+timedelta(hours=step*6)).strftime('%d July 2021 · %H:%M UTC'))
 for cursor in cursors: cursor.set_xdata([step*6,step*6])
 path=frames/f'frame_{step:02d}.png';fig.savefig(path,dpi=100)
 if step==5:
  Image.open(path).save(OUT/(SUFFIX+'_poster.png'))
 stats.append(entry);print(f'frame {step+1}/17 complete',flush=True)
plt.close(fig)
(OUT/(SUFFIX+'_metadata.json')).write_text(json.dumps({'source_header':header,'display_block_m':40,'depth_threshold_m':.1,'station':STATION,'observed_quantity':'stage' if WALF else 'discharge','simulated_quantity':'cell water depth' if WALF else 'discharge','stage_depth_datum_calibrated':False if WALF else None,'steps':stats},indent=2))
# Hold actual snapshots without synthesising intermediate flood states.
concat=OUT/(SUFFIX+'_frames.txt')
concat.write_text(''.join(f"file 'frames_{SUFFIX}/frame_{i:02d}.png'\nduration 1.5\n" for i in range(17))+f"file 'frames_{SUFFIX}/frame_16.png'\n")
base=['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-r','25','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart']
subprocess.run(base+['-crf','18',str(OUT/(SUFFIX+'_1080p.mp4'))],check=True)
subprocess.run(base+['-vf','scale=1024:576','-crf','18',str(OUT/(SUFFIX+'_TIB_1024x576.mp4'))],check=True)
print('Both videos created.',flush=True)
