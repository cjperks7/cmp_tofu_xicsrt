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
    key_diag = None,
    key_cam = None,
    config = None,
    dvol = None,
    emis_file = None,
    # HPC controls
    run_xicsrt = True,
    run_tofu = False,
    dHPC = None,
    dsave = None,
    ):

    # Init
    dout = {}

    # Prepares emissivity data
    coll = utils._prep_emis_tofu(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        emis_file=emis_file,
        #nlamb = 500,
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
            key_diag = key_diag,
            key_cam = key_cam,
            )

    # Saves XICSRT data
    if dsave is not None:
        utils._save(
            dout=dout,
            key_diag = key_diag,
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
def _run_rad_emis_xicsrt(
    coll = None,
    key_diag = None,
    key_cam = None,
    config = None,
    dvol = None,
    dHPC = None,
    ):

    # Init
    dout = {}
    lamb = coll.ddata['lamb_'+key_diag]['data']

    # Loop over wavelength
    for ii,ll in enumerate(lamb):
        # Prepares emissivity data for XICSRT
        demis = utils._prep_emis_xicsrt(
            coll = coll,
            key_diag = key_diag,
            ilamb = ii,
            dlamb = np.mean(abs(lamb[1:]-lamb[:-1]))*1e10,
            nlamb = len(lamb),
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
            lamb0 = ll
            demis = demis,
            case = 'me',
            )

        # Performs wavelength integration
        if 'signal' not in dout.keys():
            dout['signal'] = np.zeros(tmp['signal'].shape)
        dout['signal'] += tmp['signal']

        # Includes dispersion
        if 'dispersion' not in dout.keys():
            dout['dispersion'] = np.zeros(tmp['dispersion'].shape)
        dout['dispersion'] += tmp['dispersion']

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
    key_diag = None,
    key_cam = None,
    ):


    # Computes VOS
    _, coll = mv._run_mono_vol_tofu(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        key_mesh = 'm0',
        lamb = coll.ddata['lamb_'+key_diag]
        )

    # Computes signal with emissivity
    dsig = coll.compute_diagnostic_signal(
        key='flux_vos_'+key_diag,
        key_diag=key_diag,
        key_cam=[key_cam],
        key_integrand='emis_'+key_diag,
        key_ref_spectro=None,
        method='vos',
        res=None,
        mode=None,
        groupby=None,
        val_init=None,
        ref_com=None,
        brightness=False,
        spectral_binning=False,
        dvos=None,
        verb=False,
        timing=False,
        store=True,
        returnas=dict,
        )

    # Stores results
    dout = {}
    dout['signal'] = dsig[key_cam]['data'] # dim(nx,ny), [photons/bin^2]
    
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
