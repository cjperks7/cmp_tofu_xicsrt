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
    use_mag_axis_point = True, # NOTE: per the midplane LOS
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
    if use_mag_axis_point:
        tmp = utils._get_tofu_los(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            lamb0 = lamb0
            )
        dpt['ToFu']['point'] = tmp['mag_axis'] # [m], dim(3,)

    # Mesh to loop over
    dres['lambs'] = np.linspace(
        lamb0 - dres['dy'], lamb0 + dres['dy'],
        dres['ny']
        ) # dim(ny,)
    dz = np.linspace(
        dpt['ToFu']['point'][-1] - dres['dz'], 
        dpt['ToFu']['point'][-1] + dres['dz'], 
        dres['nz']
        ) # dim(nz,)
    dres['pts'] = np.array([dpt['ToFu']['point'].copy() for _ in dz]) # dim(nz,3)
    dres['pts'][:,-1] = dz # dim(nz,3)


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
    for ii, zz in enumerate(dres['pts']):
        dout['zind_%i'%(ii)] = {}

        # Alters the point source controls
        ddum = copy.deepcopy(dpt)
        ddum['ToFu']['pt'] = zz

        # Saves location
        dout['zind_%i'%(ii)]['pt'] = zz

        # Loop over wavelengths
        for jj, yy in enumerate(dres['lambs']):
            dout['zind_%i'%(ii)]['yind_%i'%(jj)] = {}

            # Runs XICSRT point source ray-tracing
            dtmp = calc._run_mono_pt_xicsrt(
                coll = coll,
                key_diag = key_diag,
                key_cam = key_cam,
                config = config,
                dpt = ddum,
                lamb0 = yy, # [AA]
                )

            # Saves wavelength data
            dout['zind_%i'%(ii)]['yind_%i'%(jj)]['lamb_AA'] = yy
            dout['zind_%i'%(ii)]['yind_%i'%(jj)]['signal'] = dtmp['signal']

    # Stores detector configuration
    dout = utils._add_det_data(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        dout = dout
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
    for ii, zz in enumerate(dres['pts']):
        dout['zind_%i'%(ii)] = {}

        # Alters the point source controls
        ddum = copy.deepcopy(dpt)
        ddum['ToFu']['pt'] = zz

        # Saves location
        dout['zind_%i'%(ii)]['pt'] = zz

        # Loop over wavelengths
        for jj, yy in enumerate(dres['lambs']):
            dout['zind_%i'%(ii)]['yind_%i'%(jj)] = {}

            # Runs ToFu point source ray-tracing
            dtmp = calc.run_mono_pt_tofu(
                coll = coll,
                key_diag = key_diag,
                key_cam = key_cam,
                dpt = ddum,
                lamb0 = yy,
                )

            # Saves wavelength data
            dout['zind_%i'%(ii)]['yind_%i'%(jj)]['lamb_AA'] = yy
            dout['zind_%i'%(ii)]['yind_%i'%(jj)]['signal'] = dtmp['signal']

    # Stores detector configuration
    dout = utils._add_det_data(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        dout = dout
        )

    # Output
    return dout

