"""One contour definition for CFD surfaces, SDF and body-coordinate masks."""
from pathlib import Path
import sys, json, numpy as np
ROOT=Path('/root/autodl-tmp/flowproxy')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'cfd/scripts')]
from flowproxy.geometry import polygon_sdf, make_grid, geometry_channels as original_geometry
import generate_case as original_case

SPECS={
 'ellipse_a1p4':dict(family='ellipse',aspect=1.4),
 'ellipse_a1p8':dict(family='ellipse',aspect=1.8),
 'rectangle_a1p3':dict(family='rectangle',aspect=1.3),
 'rectangle_a1p7':dict(family='rectangle',aspect=1.7),
 'irregular_b1':dict(family='irregular',profile=1),
 'irregular_b2':dict(family='irregular',profile=2),
}

def polygon(shape,cx=0.,cy=0.,D=1.):
 spec=SPECS[shape];family=spec['family']
 if family=='ellipse':
  t=np.linspace(0,2*np.pi,128,endpoint=False);v=np.c_[.5*spec['aspect']*np.cos(t),.5*np.sin(t)]
 elif family=='rectangle':
  a=.5*spec['aspect'];v=np.array([[-a,-.5],[a,-.5],[a,.5],[-a,.5]])
 else:
  # Four convex cubic Bezier arcs. Both profiles are symmetric about y=0,
  # but have different upstream/downstream curvature; no reflection duplicates.
  if spec['profile']==1:
   arcs=[[(.5,0),(.5,.28),(.20,.5),(-.05,.5)],
         [(-.05,.5),(-.42,.5),(-.7,.22),(-.7,0)],
         [(-.7,0),(-.7,-.22),(-.42,-.5),(-.05,-.5)],
         [(-.05,-.5),(.20,-.5),(.5,-.28),(.5,0)]]
  else:
   arcs=[[(.65,0),(.65,.20),(.35,.5),(.08,.5)],
         [(.08,.5),(-.25,.5),(-.55,.32),(-.55,0)],
         [(-.55,0),(-.55,-.32),(-.25,-.5),(.08,-.5)],
         [(.08,-.5),(.35,-.5),(.65,-.20),(.65,0)]]
  u=np.linspace(0,1,32,endpoint=False)[:,None];v=[]
  for arc in arcs:
   a,b,c,d=np.asarray(arc,float);v.append((1-u)**3*a+3*(1-u)**2*u*b+3*(1-u)*u**2*c+u**3*d)
  v=np.concatenate(v);v[:,0]-=(v[:,0].max()+v[:,0].min())/2
 v=v*D+np.array([cx,cy]);assert np.isfinite(v).all()
 area=np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1))/2
 assert area>0 and np.ptp(v[:,1])>0
 return v

def sdf(xx,yy,shape,cx=0.,cy=0.,D=1.):
 return polygon_sdf(xx,yy,polygon(shape,cx,cy,D))

def geometry_channels(cfg,shape=None,cx=None,cy=None,D=None,Re=100.,**kw):
 if shape not in SPECS:return original_geometry(cfg,shape=shape,cx=cx,cy=cy,D=D,Re=Re,**kw)
 _,_,xx,yy=make_grid(cfg);distance=sdf(xx,yy,shape,cx,cy,D).astype(np.float32);mask=(distance>0).astype(np.float32)
 walls=np.minimum(yy-cfg['domain']['y_min'],cfg['domain']['y_max']-yy).astype(np.float32)
 geom=np.stack([distance,mask,walls,np.full_like(distance,np.float32(Re/200))])
 return dict(xx=xx.astype(np.float32),yy=yy.astype(np.float32),sdf_obs=distance,mask=mask,sdf_wall=walls,geom=geom)

def write_case(cfg,out_dir,Re,cx,cy,D,shape,**kw):
 assert shape in SPECS
 case=original_case.write_case(cfg,Path(out_dir)/shape,Re,cx,cy,D,'cylinder')
 surface=case/'constant/triSurface/obstacle.stl';surface.parent.mkdir(exist_ok=True)
 z=cfg['domain']['z_span'];surface.write_text(original_case._stl_ascii(polygon(shape,cx,cy,D),-1.,z+1.).replace('solid car','solid obstacle'))
 f=case/'system/snappyHexMeshDict';text=f.read_text();start=text.index('geometry\n{');end=text.index('\ncastellatedMeshControls',start)
 text=text[:start]+'geometry\n{\n obstacle {type triSurfaceMesh; file "obstacle.stl";}\n}\n'+text[end:];f.write_text(text)
 meta=json.loads((case/'meta.json').read_text());meta.update(shape=shape,shape_spec=SPECS[shape],D_definition='cross-flow span',contour_vertices=polygon(shape,cx,cy,D).tolist());(case/'meta.json').write_text(json.dumps(meta,indent=2))
 return case

def design():
 train=[];evaluation=[];pilot=[]
 fit=[(90.,0.,1.),(120.,.3,1.1),(160.,-.3,1.05),(190.,0.,1.15)]
 check=[(107.3,.137,1.08),(176.7,-.487,1.18)]
 for index,shape in enumerate(SPECS):
  for role,settings,dest in [('newshape_train',fit,train),('newshape_evaluation',check,evaluation)]:
   for Re,cy,D in settings:
    q=dict(id=f'{shape}_Re{Re:g}_cy{cy:g}_D{D:g}',shape=shape,Re=Re,cx=0.,cy=cy,D=D,role=role,family=SPECS[shape]['family'])
    dest.append(q)
    if role=='newshape_train' and (Re==120. if index%2==0 else Re==160.):pilot.append(q)
 assert len(train)==24 and len(evaluation)==12 and len(pilot)==6
 return dict(train=train,evaluation=evaluation,pilot=pilot,shape_specs=SPECS)
