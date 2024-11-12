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
    method = 'camera_midline', # 'vos_midline'
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
    if method == 'camera_midline':
        indy = int(lamb.shape[1]/2-1) # Takes midline
        #indx = np.nanargmin(abs(lamb0*1e-10 - lamb[:,indy]))

        # If array is monotonically decreasing
        if np.nanmean(lamb[:,indy][1:]-lamb[:,indy][:-1]) <0:
            indx_up = np.where(lamb[:,indy]>lamb0*1e-10)[0][-1] # last greater number
            indx_low = np.where(lamb[:,indy]<=lamb0*1e-10)[0][0] # first smaller number
        # If array is monotonically increasing
        if np.nanmean(lamb[:,indy][1:]-lamb[:,indy][:-1]) >0:
            indx_up = np.where(lamb[:,indy]>lamb0*1e-10)[0][0] # first greater number
            indx_low = np.where(lamb[:,indy]<=lamb0*1e-10)[0][-1] # last smaller number

    elif method == 'vos_midline':
        indy = 0
        res0 = 1e5
        for yy in np.arange(lamb.shape[1]):
            if np.all(np.isnan(lamb[:,yy])):
                continue
            # Find LOS that terminates in the middle vertically of VOS for this wavelength
            else:
                #indx = np.nanargmin(abs(lamb0*1e-10 - lamb[:,yy]))

                #res = abs(
                #    ptsz[-1,indx,yy] - np.nanmean(pcross1)
                #    )
                # If array is monotonically decreasing
                if np.nanmean(lamb[:,indy][1:]-lamb[:,indy][:-1]) <0:
                    indx_up = np.where(lamb[:,indy]>lamb0*1e-10)[0][-1] # last greater number
                    indx_low = np.where(lamb[:,indy]<=lamb0*1e-10)[0][0] # first smaller number
                # If array is monotonically increasing
                if np.nanmean(lamb[:,indy][1:]-lamb[:,indy][:-1]) >0:
                    indx_up = np.where(lamb[:,indy]>lamb0*1e-10)[0][0] # first greater number
                    indx_low = np.where(lamb[:,indy]<=lamb0*1e-10)[0][-1] # last smaller number

                res_up = abs(
                    ptsz[-1,indx_up,yy] - np.nanmean(pcross1)
                    )
                res_low = abs(
                    ptsz[-1,indx_low,yy] - np.nanmean(pcross1)
                    )
                res = 0.5*(res_up+res_low)

                if res < res0 and not np.isnan(res):
                    res0 = res.copy()
                    indy = yy.copy()

    # Interpolates to get LOS of interest
    p_in_up = np.r_[
        ptsx[-1,indx_up,indy], ptsy[-1,indx_up,indy], ptsz[-1,indx_up,indy]
        ] # Inboard wall
    p_in_low = np.r_[
        ptsx[-1,indx_low,indy], ptsy[-1,indx_low,indy], ptsz[-1,indx_low,indy]
        ] # Inboard wall
    p_cry_up = np.r_[
        ptsx[-2,indx_up,indy], ptsy[-2,indx_up,indy], ptsz[-2,indx_up,indy]
        ] # Inboard wall
    p_cry_low = np.r_[
        ptsx[-2,indx_low,indy], ptsy[-2,indx_low,indy], ptsz[-2,indx_low,indy]
        ] # Inboard wall

    ratio = (lamb0*1e-10 - lamb[indx_low,indy])/(lamb[indx_up,indy] - lamb[indx_low,indy])
    p_in = p_in_low + ratio *(p_in_up-p_in_low)
    p_cry = p_cry_low + ratio *(p_cry_up-p_cry_low)

    # Get LOS vector
    vect = p_in - p_cry
    vect /= np.linalg.norm(vect)

    # Finds when LOS is closest to magnetic axis
    tt = np.linspace(0,2,501) # [m]
    ps = p_in[None,:] - tt[:,None]*vect[None,:] # dim(npt, 3)
    rs = np.sqrt(ps[:,0]**2 + ps[:,1]**2) # dim(npt,)

    indr = np.argmin(abs(rs-R0))
    p_ma = p_in - tt[indr]*vect

    # Distance from magnetic axis point to crystal
    L_ma2cry = np.linalg.norm(p_ma-p_cry) # [m]

    # Output
    return { 
        'los_vect': vect, # LOS vector
        'los_inboard': p_in, # [m], Inboard wall location
        'los_mag_axis': p_ma, # [m], Location closest to magnetic axis
        'los_cry': p_cry, # [m], Location on crystal surface
        'L_ma2cry': L_ma2cry, # [m], Distance from mag. axis pt. to crystal surface
        }
    
    