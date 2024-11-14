'''

Script to handle calculating the spatial- and spectral-
resolution of a spectrometer using monochromatic point
sources distributed around a wavelength about the magnetic axis

cjperks
Nov 7th, 2024

'''

# Modules
import xicsrt

import numpy as np
import matplotlib.pyplot as plt
import copy

import cmp_tofu_xicsrt.utils as utils
import cmp_tofu_xicsrt.setup._def_point as dp
import cmp_tofu_xicsrt.calc as calc

__all__ = [
    'run_resolution'
    ]

###################################################
#
#           Main
#
###################################################

# Simulates a distribution of point sources ...
def run_resolution(
    coll = None,
    key_diag = None,
    key_cam = None,
    config = None,
    lamb0 = None,
    cry_shape = None,
    dpt = None,
    dres = None,
    los_method = 'camera_midline',
    etendue_degree = 2,
    # HPC controls
    run_xicsrt = None,
    run_tofu = None,
    dsave = None,
    ):

    # Init
    dout = {}

    # Gets default values
    if dpt is None:
        dpt = dp.get_dpt(option=key_diag)
    if dres is None:
        dres = dp.get_dres(option=key_diag)

    # If user speficially wants pt to be on the magnetic axis
    # NOTE: per the midplane LOS
    danchor = utils._get_tofu_los(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        lamb0 = lamb0,
        method = los_method
        )
    dpt['ToFu']['point'] = danchor['los_mag_axis'] # [m], dim(3,)

    # Gets dot product between LOS vector and aperture normal
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    kap = doptics[key_cam]['optics'][1:][0]
    cos_anchor = np.dot(
        danchor['los_vect'], 
        coll.dobj['aperture'][kap]['dgeom']['nin']
        )

    # Mesh to loop over
    dres['lambs'] = np.linspace(
        lamb0 - dres['dy'], lamb0 + dres['dy'],
        dres['ny']
        ) # dim(ny,)
    dz = np.linspace(
        -dres['dz'], dres['dz'],
        dres['nz']
        ) # dim(nz,)

    # Init
    dres['pts'] = np.zeros(dz.shape + dres['lambs'].shape + (3,)) # dim(nz,ny,3)

    # Loop over vertical scan
    for ii, zz in enumerate(dz):
        # Loop over spectral scan
        for jj, yy in enumerate(dres['lambs']):
            # Get LOS for this wavelength
            dnew = utils._get_tofu_los(
                coll = coll,
                key_diag = key_diag,
                key_cam = key_cam,
                lamb0 = yy,
                method = los_method
                )

            # Find LOS distance to keep etendue roughly the same
            cos_new = np.dot(
                dnew['los_vect'], 
                coll.dobj['aperture'][kap]['dgeom']['nin']
                )
            L_new = danchor['L_ma2cry'] *(
                cos_new/cos_anchor
                )**(1/etendue_degree)

            # Defines point
            dres['pts'][ii,jj,:] = (
                dnew['los_cry']
                + L_new * dnew['los_vect']
                )

        # Adds the vertical shift
        dres['pts'][ii,:,-1] += zz


    # Runs XICSRT
    if run_xicsrt:
        dout['XICSRT'] = _run_res_pt_xicsrt(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            config = config,
            dpt = dpt,
            dres = dres,
            lamb0 = lamb0,
            )

    # Runs ToFu
    if run_tofu:
        dout['ToFu'] = _run_res_pt_tofu(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            dpt = dpt,
            dres = dres,
            lamb0 = lamb0
            )

    # Saves XICSRT data
    if dsave is not None:
        utils._save(
            dout = dout,
            case = 'pt',
            lamb0 = lamb0,
            dsave = dsave,
            )

    # Output
    return dout

###################################################
#
#           Utilities
#
###################################################

# ... in XICSRT
def _run_res_pt_xicsrt(
    coll = None,
    key_diag = None,
    key_cam = None,
    config = None,
    dpt = None,
    dres = None,
    lamb0 = None,
    ):

    # Init
    dout = {}

    # Loop over vertical locations
    for ii in np.arange(dres['pts'].shape[0]):
        dout['zind_%i'%(ii)] = {}

        # Loop over wavelengths
        for jj, yy in enumerate(dres['lambs']):
            # Alters the point source controls
            ddum = copy.deepcopy(dpt)
            ddum['ToFu']['point'] = dres['pts'][ii,jj,:]

            dout['zind_%i'%(ii)]['yind_%i'%(jj)] = {}

            # Runs XICSRT point source ray-tracing
            dtmp = calc._run_mono_pt_xicsrt(
                coll = coll,
                key_diag = key_diag,
                key_cam = key_cam,
                config = config,
                dpt = ddum,
                lamb0 = yy, # [AA]
                nx = 1028,
                ny = 1062,
                )

            # Saves wavelength data
            dout['zind_%i'%(ii)]['yind_%i'%(jj)]['lamb_AA'] = yy
            dout['zind_%i'%(ii)]['yind_%i'%(jj)]['signal'] = dtmp['signal']

            # Saves location data
            dout['zind_%i'%(ii)]['yind_%i'%(jj)]['pt'] = dres['pts'][ii,jj,:]

    # Stores detector configuration
    dout['signal'] = np.zeros_like(dtmp['signal']) # dummy value to get function to work
    dout = utils._add_det_data(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        dout = dout,
        split = True,
        )

    # Output
    return dout

# ... in ToFu
def _run_res_pt_tofu(
    coll = None,
    key_diag = None,
    key_cam = None,
    dpt = None,
    dres = None,
    lamb0 = None
    ):

    # Init
    dout = {}

    # Loop over vertical locations
    for ii in np.arange(dres['pts'].shape[0]):
        dout['zind_%i'%(ii)] = {}

        # Loop over wavelengths
        for jj, yy in enumerate(dres['lambs']):
            # Alters the point source controls
            ddum = copy.deepcopy(dpt)
            ddum['ToFu']['point'] = dres['pts'][ii,jj,:]

            dout['zind_%i'%(ii)]['yind_%i'%(jj)] = {}

            # Runs ToFu point source ray-tracing
            dtmp = calc._run_mono_pt_tofu(
                coll = coll,
                key_diag = key_diag,
                key_cam = key_cam,
                dpt = ddum,
                lamb0 = yy,
                )

            # Saves wavelength data
            dout['zind_%i'%(ii)]['yind_%i'%(jj)]['lamb_AA'] = yy
            dout['zind_%i'%(ii)]['yind_%i'%(jj)]['signal'] = dtmp['signal']

            # Saves location data
            dout['zind_%i'%(ii)]['yind_%i'%(jj)]['pt'] = dres['pts'][ii,jj,:]

    # Stores detector configuration
    dout['signal'] = np.zeros_like(dtmp['signal']) # dummy value to get function to work
    dout = utils._add_det_data(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        dout = dout,
        split = False,
        )

    # Output
    return dout

