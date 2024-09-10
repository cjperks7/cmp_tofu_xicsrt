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
    key_diag = None,
    key_cam = None,
    config = None,
    dvol = None,
    lamb0 = None,
    # HPC controls
    run_xicsrt = True,
    run_tofu = False,
    dHPC = None,
    dsave = None,
    ):

    # Init
    dout = {}

    # Adds mesh data to compute VOS on
    coll = utils._add_mesh_data(
        coll = coll,
        case = 'simple',
        lamb = lamb0
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
            key_diag = key_diag,
            key_cam = key_cam,
            lamb0 = lamb0
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
        '''
        dout = _calc_signal(
            rays = tmp['detector']['origin'],
            voxels = tmp['voxels'],
            config = config,
            dout = dout,
            fE = fE[ii],
            dlamb = lamb[1]-lamb[0],
            nlamb = len(lamb),
            ilamb = ii,
            )
        '''
        dout['voxels'] = tmp['voxels']
        dout = utils._calc_signal(
            dout = dout,
            config = config,
            det_origin = tmp['detector']['origin'],
            dlamb = lamb[1]-lamb[0],
            ilamb = ilamb,
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
    key_diag = None,
    key_cam = None,
    lamb0 = None,
    ):

    # Builds wavelength mesh, [AA], [1/AA], dim(nlamb,)
    lamb, fE = utils._build_gaussian(lamb0=lamb0)

    # Prepares Gaussian emissivity at 1 ph/s/cm3
    emiss = (
        1e-6 # [ph/s/m3]
        *fE*1e10 # [1/m]
        / (4*np.pi) # [1/sr]
        ) # [ph/s/m3/sr/m], dim(nlamb,)

    # Assumes spatially homogeneous
    nR, nZ = coll.dobj['mesh']['mRZ']['shape-k']
    emissRZ = np.repeat(
        np.repeat(emiss[None, None, :], nR, axis=0),
        nZ,
        axis=1,
        )

    # Adds data to collection object
    #coll.add_mesh_1d(
    #    key='mlamb',
    #    knots=lamb*1e-10,
    #    deg=1,
    #    units='m',
    #    )

    eunit = 'ph/s/m3/sr/m'
    coll.add_data(
        key='emiss',
        data=emiss,
        ref='mlamb',
        units=eunit,
        )
    coll.add_data(
        key='emissRZ',
        data=emissRZ,
        ref=('mRZ', 'mlamb'),
        units=eunit,
        )

    # Computes VOS
    _, coll = mv._run_mono_vol_tofu(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        lamb0 = lamb0, # [AA]
        )

    # Computes signal with emissivity
    dsig = coll.compute_diagnostic_signal(
        key='synth_vos_interp',
        key_diag=key_diag,
        key_cam=None,
        key_integrand='emissRZ',
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


'''
###################################################
#
#           Utilities
#
###################################################

# Builds wavelength mesh
def _build_lamb(
    lamb0 = None, # [AA], centroid
    slamb = 5e-4, # [AA], std
    nlamb = 61, # num grid points
    xlamb = 3.2, # num of std's
    ):

    # Wavelength mesh, [AA]
    lamb = np.linspace(
        lamb0 - xlamb*slamb,
        lamb0 + xlamb*slamb,
        nlamb
        )

    # Normalized distribution, [1/AA]
    fE = (
        1/(slamb *np.sqrt(2*np.pi))
        *np.exp(-0.5*(lamb-lamb0)**2/slamb**2)
        )

    # Error check
    if False:
        print((lamb[1]-lamb[0])/lamb0*100)
        print((1-np.trapz(fE, lamb))*100)
        
        fig, ax = plt.subplots()
        ax.plot(lamb, fE, '*')

    # Output, [AA], [1/AA], dim(nlamb,)
    return lamb, fE

# Performs wavelength-integration of VOS
def _calc_signal(
    rays = None,
    config = None,
    voxels = None,
    dout = None,
    fE = None,
    dlamb = None,
    nlamb = None,
    ilamb = None,
    nx = 512,
    ny = 256,
    ):

    # Calculates histogram
    dhist = utils._calc_det_hist(
        rays = rays,
        config = config,
        nx = nx,
        ny = ny,
        ) # dim(hor_pix, vert_pix)

    # Init
    if 'signal' not in list(dout.keys()):
        dout['signal'] = np.zeros((nx,ny))
    if 'dispersion' not in list(dout.keys()):
        dout['dispersion'] = np.zeros((nx,ny,nlamb))

    # Stores dispersion histogram
    dout['dispersion'][:,:,ilamb] += dhist['counts'] 

    # wavelength-integration (via left-hand Riemann sum)
    dout['signal'] += (
        dhist['counts'] 
        * fE * dlamb
        / utils._conv_2normEmis(voxels=voxels, case='me')
        ) # dim(nx,ny), [photons/bin^2]

    # Output
    return dout
'''