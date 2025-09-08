'''

Script to handle simulating a multi-energy, volume source

'''

# Modules
import xicsrt
import tofu as tf

import numpy as np
import matplotlib.pyplot as plt
import sys, os

import cmp_tofu_xicsrt.utils as utils
import cmp_tofu_xicsrt.calc._mono_vol as mv


__all__ = [
    'run_multi_vol',
    ]

##############################################
#
#           Main
#
##############################################

# Simulates a multi-energy, volume source ...
def run_multi_vol(
    coll = None,
    lamb0 = None,
    key_diag = None, key_cam = None,
    key_mesh = None, key_lamb = None, key_emis = None,
    config = None,
    subcam = None,
    # Velocity controls
    add_velocity = False, dvel = None,
    # HPC controls
    run_xicsrt = True, run_tofu = False,
    dHPC = None,
    dsave = None,
    dvol = None,
    dvos_tf = None,
    ):

    # Init
    dout = {}

    # Adds mesh data to compute VOS on
    coll = utils._prep_emis_tofu(
        coll = coll,
        lamb0 = lamb0,
        case = 'me',
        key_mesh = key_mesh, key_lamb = key_lamb, key_emis = key_emis,
        )

    # Runs XICSRT
    if run_xicsrt:
        dout['XICSRT'] = _run_multi_vol_xicsrt(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            config = config,
            dvol = dvol,
            dHPC = dHPC,
            lamb0 = lamb0
            )

    # Runs ToFu
    if run_tofu:
        dout['ToFu'] = _run_multi_vol_tofu(
            coll = coll,
            key_diag = key_diag, key_cam = key_cam,
            key_mesh = key_mesh, key_lamb = key_lamb, key_emis = key_emis,
            dvos_tf = dvos_tf,
            subcam = subcam,
            )

    # Saves XICSRT data
    if dsave is not None:
        utils._save(
            dout=dout,
            lamb0 = lamb0,
            dsave = dsave,
            dHPC=dHPC,
            case = 'me',
            )
    
    # Output
    return dout

###################################################
#
#           Simulations
#
###################################################

# ... in XICSRT
def _run_multi_vol_xicsrt(
    coll = None,
    key_diag = None,
    key_cam = None,
    config = None,
    dvol = None,
    dHPC = None,
    #vol_plt = vol_plt,
    lamb0 = None
    ):

    # Init
    dout = {}

    # Builds wavelength mesh
    lamb, fE = utils._build_gaussian(lamb0=lamb0)

    # Loop over wavelength
    for ii,ll in enumerate(lamb):

        # Calculates VOS-int per wavelength
        tmp = mv._run_mono_vol_xicsrt(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            config = config,
            dvol = dvol,
            dHPC = dHPC,
            calc_signal = False,
            lamb0 = ll
            )

        # Performs wavelength integration
        dout['voxels'] = tmp['voxels']
        dout = utils._calc_signal(
            dout = dout,
            config = config,
            det_origin = tmp['detector']['origin'],
            dlamb = lamb[1]-lamb[0],
            ilamb = ii,
            nlamb = len(lamb),
            emis_val = fE[ii],
            case = 'me',
            )

    # Stores detector configuration
    dout = utils._add_det_data(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        dout = dout
        )
    dout['lambda_AA'] = lamb # dim(nlamb,)

    # Output
    return dout

# ... in ToFu
def _run_multi_vol_tofu(
    coll = None,
    key_diag = None, key_cam = None,
    key_mesh = None, key_lamb = None, key_emis = None,
    # Volume discretization controls
    dvos_tf = None,
    # Other
    subcam = None,
    ):

    # Calculates VOS matrix if neccessary
    if dvos_tf['run_vos']:
        coll = utils._compute_vos_tofu(
            coll = coll,
            key_diag = key_diag,
            key_mesh = key_mesh, key_lamb = key_lamb,
            dvos_tf = dvos_tf
            )

    # Calculates signal
    dsig = utils._compute_signal_tofu(
        coll = coll,
        key_diag = key_diag, key_cam = key_cam,
        key_emis = key_emis,
        )

    # Stores results
    dout = {}
    dout['signal'] = dsig[key_cam]['data'] # dim(nx,ny), [photons/s/bin^2]
    
    # Stores detector configuration
    dout = utils._add_det_data(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        dout = dout,
        split = False,
        subcam = subcam,
        )

    # Output
    return dout