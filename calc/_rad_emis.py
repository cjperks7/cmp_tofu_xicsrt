'''

Script to handle simulating radially peaked emissivity data

cjperks
Sep 9th, 2024

'''

# Modules
import xicsrt
import tofu as tf

import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.interpolate import griddata, interp1d

import cmp_tofu_xicsrt.utils as utils
import cmp_tofu_xicsrt.calc._mono_vol as mv

__all__ = [
    'run_rad_emis'
]

##############################################
#
#           Main
#
##############################################

# Simulates radially-peaked emissivity data...
def run_rad_emis(
    coll = None,
    emis_file = None,
    key_diag = None, key_cam = None,
    key_mesh = None, key_lamb = None, key_emis = None,
    config = None,
    # HPC controls
    run_xicsrt = True, run_tofu = False,
    dHPC = None,
    dvol = None,
    dvos_tf = None,
    dsave = None,
    ):

    # Init
    dout = {}

    # Prepares emissivity data
    coll = utils._prep_emis_tofu(
        coll = coll,
        emis_file = emis_file,
        case = 'rad_emis',
        key_diag = key_diag, #key_cam = key_cam,
        key_mesh = key_mesh, key_lamb = key_lamb, key_emis = key_emis,
        nlamb = dHPC['nlamb'],
        )

    # Runs XICSRT
    if run_xicsrt:
        dout['XICSRT'] = _run_rad_emis_xicsrt(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            config = config,
            dvol = dvol,
            dHPC = dHPC,
            )

    # Runs ToFu
    if run_tofu:
        dout['ToFu'] = _run_rad_emis_tofu(
            coll = coll,
            key_diag = key_diag, key_cam = key_cam,
            key_mesh = key_mesh, key_lamb = key_lamb, key_emis = key_emis,
            dvos_tf = dvos_tf,
            )

    # Saves XICSRT data
    if dsave is not None:
        utils._save(
            dout=dout,
            key_diag = key_diag,
            dsave = dsave,
            dHPC=dHPC,
            case = 'emis',
            )
    
    # Output
    return dout


###################################################
#
#           Simulations
#
###################################################

# ... in XICSRT
def _run_rad_emis_xicsrt(
    coll = None,
    key_diag = None,
    key_cam = None,
    config = None,
    dvol = None,
    dHPC = None,
    # Histogram size
    nx = 1028,
    ny = 1062,
    ):

    # Init
    dout = {}
    lamb_lim = dHPC['lamb_lim']
    lamb_num = dHPC['lamb_num']     # Takes this specfic wavelength group
    lamb = coll.ddata['mlamb_'+key_diag+'_k']['data'][lamb_num*lamb_lim:(lamb_num+1)*lamb_lim]*1e10 # [AA]

    # Loop over wavelength
    for ii,ll in enumerate(lamb):
        print('Lamb step')
        print(ii)
        print(ll)
        # Prepares emissivity data for XICSRT
        demis = utils._prep_emis_xicsrt(
            coll = coll,
            key_diag = key_diag,
            ilamb = ii + lamb_num*lamb_lim,
            #ilamb = np.nanargmin(abs(coll.ddata['mlamb_'+key_diag+'_k']['data'] - ll*1e-10)),
            dlamb = np.mean(abs(lamb[1:]-lamb[:-1])), # [AA]
            nlamb = len(coll.ddata['mlamb_'+key_diag+'_k']['data']),
            )

        # Calculates VOS-int per wavelength
        tmp = mv._run_mono_vol_xicsrt(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            config = config,
            dvol = dvol,
            dHPC = dHPC,
            calc_signal = True,
            lamb0 = ll, # [AA]
            demis = demis,
            #case = 'me',
            case = 'mv',
            nx = nx,
            ny = ny,
            )

        # Performs wavelength integration
        if 'signal' not in dout.keys():
            dout['signal'] = np.zeros(tmp['signal'].shape)
        dout['signal'] += tmp['signal']

        # Includes dispersion
        #if 'dispersion' not in dout.keys():
        #    dout['dispersion'] = np.zeros(tmp['dispersion'].shape)
        #dout['dispersion'] += tmp['dispersion']

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
def _run_rad_emis_tofu(
    coll = None,
    key_diag = None, key_cam = None,
    key_mesh = None, key_lamb = None, key_emis = None,
    dvos_tf = None,
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
        )

    # Output
    return dout
