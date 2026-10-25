"""Geometry-only quadratic moving-least-squares decoder to the physical grid."""
import numpy as np
from scipy.spatial import cKDTree

def decoder(xx, yy, mask, domain, nquery=64, nneighbor=16):
    x=np.linspace(domain['x_min'],domain['x_max'],nquery)
    y=np.linspace(domain['y_min'],domain['y_max'],nquery)
    qx,qy=np.meshgrid(x,y);query=np.stack([qx.ravel(),qy.ravel()],1)
    physical=np.stack([xx.ravel(),yy.ravel()],1).astype(np.float64)
    valid=np.flatnonzero(mask.ravel()>.5)
    distance,index=cKDTree(physical[valid]).query(query,k=nneighbor)
    donors=valid[index];offset=physical[donors]-query[:,None]
    scale=np.maximum(distance[:,-1],1e-8)
    z=offset/scale[:,None,None];dx=z[:,:,0];dy=z[:,:,1]
    design=np.stack([np.ones_like(dx),dx,dy,dx*dx,dx*dy,dy*dy],-1)
    wt=1/((distance/scale[:,None])**2+.05**2);wt/=wt.sum(1,keepdims=True)
    matrix=np.einsum('nki,nk,nkj->nij',design,wt,design)
    matrix+=np.diag([0,1e-7,1e-7,1e-7,1e-7,1e-7])[None]
    rhs=np.zeros((len(query),6));rhs[:,0]=1
    solved=np.linalg.solve(matrix,rhs[:,:,None])[:,:,0]
    coef=np.einsum('ni,nki,nk->nk',solved,design,wt)
    coef/=coef.sum(1,keepdims=True)
    exact=distance[:,0]<1e-7
    coef[exact]=0;coef[exact,0]=1
    if not np.isfinite(coef).all():raise ValueError('Non-finite reconstruction coefficients')
    if np.max(abs(coef))>5:raise ValueError('Unstable reconstruction coefficient')
    # A deterministic geometry-only sanity check, independent of all flow fields.
    basis=np.stack([np.ones(len(physical)),physical[:,0],physical[:,1]],-1)
    reproduced=np.einsum('nk,nkc->nc',coef,basis[donors])
    desired=np.stack([np.ones(len(query)),query[:,0],query[:,1]],-1)
    return donors.astype(np.int64),coef.astype(np.float32),{
        'neighbors':nneighbor,'max_abs_coefficient':float(abs(coef).max()),
        'constant_linear_physical_field_max_delta':float(abs(reproduced-desired).max()),
        'degree':2,'ridge_on_nonconstant_terms':1e-7,'fluid_donors_only':True,
        'geometry_only_no_flow_value_fitting':True}

def apply_numpy(fields, donors, weights):
    """fields [T,3,H,W], output [T,3,64,64]."""
    flat=fields.reshape(len(fields),3,-1)
    values=flat[:,:,donors]
    out=np.einsum('tcnk,nk->tcn',values,weights)
    return out.reshape(len(fields),3,64,64)
