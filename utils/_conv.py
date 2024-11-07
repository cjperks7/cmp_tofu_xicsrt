'''

Functions to convert between XICSRT and ToFu coordinate representation

cjperks
Aug 5, 2024

'''

# Modules
import numpy as np

import tofu as tf

__all__ = [
    '_tofu2xicsrt',
    '_xicsrt2tofu',
    '_get_tofu_los'
    ]



# Function to convert the coordinate basis from ToFu to XICSRT
def _tofu2xicsrt(
    data = None,
    ):
    '''
    In comparing the two codes, it is important to unify the global coordinate
    basis when defining sources off the axis of symmetry of the diagnostic

    Consider a point or vector defined by coordinates np.r_[x,y,z] with basis
    x = np.r_[1,0,0], y = np.r_[0,1,0], and z = np.r_[0,0,1]

    We will assume the real-world SPARC diagnostic geometry as good-truth, so
    the origin is the center of the tokamak and looking top-down the diagnostic
    hall is to the East

    In ToFu:
        x -> East
        y -> North
        z -> Up

    In XICSRT:
        x -> North
        y -> Up
        z -> East

    This amounts to two 90degree rotations of the basis with no flipping, so
    doing this transformation does nothing to change the physical results but
    is more "natural" given the terminology employed in XICSRT

    '''

    # Output
    if np.ndim(data) == 1:
        return np.r_[data[1], data[2], data[0]]
    elif np.ndim(data) == 2:
        return np.vstack(
            (data[:,1],
            data[:,2],
            data[:,0]
            )).T

# Function to convert the coordinate basis from XICSRT to ToFu
def _xicsrt2tofu(
    data = None,
    ):
    if np.ndim(data) == 1:
        return np.r_[data[2], data[0], data[1]]
    elif np.ndim(data) == 2:
        return np.vstack(
            (data[:,2],
            data[:,0],
            data[:,1]
            )).T

##################################################
#
#               Other
#
##################################################

# Function to get LOS-per-wavelength from ToFu
def _get_tofu_los(
    coll = None,
    key_diag = None,
    key_cam = None,
    lamb0 = None,
    R0 = 1.89553,      # [m], magnetic axis (with Shafarnov shift)
    ):

    # Gets wavelength mesh
    lamb, refs = coll.get_diagnostic_lamb(
        key_diag,
        key_cam=key_cam,
        lamb='lamb',
        ) # dim(nx, ny)

    # Gets LOS point/vector data
    ptsx, ptsy, ptsz = coll.get_rays_pts(key_diag)
    vx, vy, vz = coll.get_rays_vect(key_diag)

    # extract keys to R, Z coordinates of polygon definign vos in poloidal cross-section
    pcross0, pcross1 = tf.data._class8_vos_utilities._get_overall_polygons(
        coll, 
        doptics=coll.dobj['diagnostic'][key_diag]['doptics'], 
        key_cam=key_cam, 
        poly='pcross', 
        convexHull=False
        )
    
    # Finds indices of interest
    #indy = int(lamb.shape[1]/2-1) # Takes midline
    #indy = np.nanargmin(abs(
    #    ptsz[-1,indx,:] - np.nanmean(ptsz[-1,indx,:])
    #    ))
    #indx = np.nanargmin(abs(lamb0*1e-10 - lamb[:,indy]))

    indy = 0
    res0 = 1e5
    for yy in np.arange(lamb.shape[1]):
        if np.all(np.isnan(lamb[:,yy])):
            continue
        # Find LOS that terminates in the middle vertically of VOS for this wavelength
        else:
            indx = np.nanargmin(abs(lamb0*1e-10 - lamb[:,yy]))

            res = abs(
                ptsz[-1,indx,yy] - np.nanmean(pcross1)
                )

            if res < res0 and not np.isnan(res):
                res0 = res.copy()
                indy = yy.copy()

    # Gets LOS of interest
    vect = np.r_[vx[-1,indx,indy], vy[-1,indx,indy], vz[-1,indx,indy]]
    p0 = np.r_[ptsx[-1,indx,indy], ptsy[-1,indx,indy], ptsz[-1,indx,indy]] # Inboard wall

    # Finds when LOS is closest to magnetic axis
    tt = np.linspace(0,2,501) # [m]
    ps = p0[None,:] - tt[:,None]*vect[None,:] # dim(npt, 3)
    rs = np.sqrt(ps[:,0]**2 + ps[:,1]**2) # dim(npt,)

    indr = np.argmin(abs(rs-R0))

    # Output
    return {
        'los_vect': vect, # LOS vector
        'los_p0': p0, # Inboard wall location
        'mag_axis': p0 - tt[indr]*vect # Location closest to magnetic axis
        }
    
    