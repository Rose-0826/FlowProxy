from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path('/root/autodl-tmp/flowproxy');sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_evaluate_v1 import CFG
from geometry_attack_v2 import NativeSampler
from cpu_data_prepare import frames
from flowproxy.geometry import geometry_channels
OUT=ROOT/'outputs/aic_showcase_2026-10-02/animations';OUT.mkdir(parents=True,exist_ok=True)
manifest=[]
for role,cid,Re,cy in [('median','square_Re170_cy0.7_D1',170,.7),('worst_D1','square_Re110_cy-0.7_D1',110,-.7)]:
 folder='geometry_attack_confirmation_far_v1';items=frames(ROOT/'data/raw'/folder/cid);geo=geometry_channels(CFG,shape='square',Re=Re,cx=0,cy=cy,D=1);m=geo['mask']>.5;sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask'])
 truth=np.stack([sampler.sample(f)[1] for _,f in items[4:164]])
 kfile=ROOT/'outputs'/folder/'evaluation/predictions'/f'K8__{cid}.npz';hfile=ROOT/'outputs'/folder/'evaluation/predictions'/f'neural_periodic_closure__{cid}.npz'
 K=np.load(kfile)['prediction'][:,1];H=np.load(hfile)['prediction'][:,1]
 lim=max(float(np.max(np.abs(a[:,m]))) for a in [truth,K,H]);ids=list(range(0,160,5))+[159];pictures=[];cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#cbd5df')
 for i in ids:
  fig=plt.figure(figsize=(12,3.2),dpi=100,facecolor='white');im=None
  for col,(title,a) in enumerate([('CFD reference',truth),('K8 rollout',K),('Short trajectory + cycle',H)]):
   ax=fig.add_axes([.015+.316*col,.22,.29,.64]);im=ax.imshow(np.ma.masked_where(~m,a[i]),origin='lower',extent=[-8,20,-8,8],interpolation='nearest',cmap=cmap,vmin=-lim,vmax=lim,aspect='equal');ax.set_axis_off();ax.set_title(title,fontsize=16,pad=10)
  cb=fig.colorbar(im,cax=fig.add_axes([.943,.25,.012,.52]),ticks=[-lim,0,lim]);cb.ax.tick_params(labelsize=11);cb.ax.set_title('Uy',fontsize=12)
  e=lambda a:np.linalg.norm((a[i]-truth[i])[m])/max(np.linalg.norm(truth[i][m]),1e-8)*100
  fig.text(.02,.065,f'{cid}  |  t = {items[i+4][0]:.2f} s  |  Uy error: K8 {e(K):.2f}% ; cycle {e(H):.2f}%',fontsize=14,color='#102A43')
  fig.canvas.draw();arr=np.asarray(fig.canvas.buffer_rgba());pictures.append(Image.fromarray(arr[:,:,:3].copy()));plt.close(fig)
 file=OUT/f'{role}_40s_Uy.gif';pictures[0].save(file,save_all=True,append_images=pictures[1:],duration=250,loop=0,optimize=False)
 manifest.append({'case':cid,'role':role,'role_rule':'lower median R among original new D1 eight' if role=='median' else 'worst R among original new D1 eight','file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'source_prediction_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [kfile,hfile]},'physical_frame_times':[items[i+4][0] for i in ids],'selected_future_indices':ids,'no_time_interpolation':True,'native_64x64_full_domain_no_spatial_smoothing':True,'common_color_limits_all_times_and_models':[-lim,lim],'display_frame_duration_ms':250,'playback_is_faster_than_physical_time':True})
 print('ANIMATION_CREATED',role,file.stat().st_size,flush=True)
(OUT/'manifest.json').write_text(json.dumps({'animations':manifest,'not_a_full160frame_video':True},indent=2))
