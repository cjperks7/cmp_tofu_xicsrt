'''

XICSRT_utils is a function meant to manage various comparison
analyzes between XICSRT and ToFu

cjperks
Mar 19th, 2024

'''

# Modules
import os, sys
import numpy as np
import copy

import cmp_tofu_xicsrt.calc as calc
import cmp_tofu_xicsrt.setup as setup


__all__ = [
    'main',
    '_save',
    '_get_mv_results'
    ]

###################################################
#
#           Main
#
###################################################

# Manages the type of XICSRT analyzes to do
def main(
    # ToFu data
    lamb0=1.61, # [AA]
    coll_tf = None,
    key_diag = 'valid', key_cam = 'cam',
    key_mesh = 'mRZ', key_lamb = 'mlamb', key_emis = 'emis',
    subcam = None,
    doptics = None,
    # XICSRT data
    cry_shape = 'Spherical',
    niter = 5,
    # Monochromatic, point source controls
    pt_run = False, dpt = None,     # Resolution controls
    # Monochromatic, volumetric source controls
    vol_run = False, dvol = None,    # Resolution controls
    # Multi-energy, volumetric source controls
    me_run = False,
    # Radially-peaked emissivity data
    rad_run = False, emis_file = None,
    # Spectral-/spatial-resolution data
    res_run = False, dres = None,
    # Velocity control
    add_velocity = False, dvel = None,
    # HPC controls
    run_xicsrt = True, run_tofu = False,
    dHPC = None,
    dsave = None,
    dvos_tf = {
        'run_vos': True,
        'res_RZ': [0.01, 0.01],
        'res_phi': 0.0005,
        'n0': 181,
        'n1': 101,
        'save': False,
        'path': '/home/cjperks/orcd/scratch/work/tofu_sparc/diags',
        },     # Controls for TOFU VOS computation
    ):

    # Builds ToFu diagnostic
    coll = setup._init_diag(
        coll = coll_tf,
        dcry = cry_shape,
        lamb0 = lamb0,
        subcam = subcam,
        doptics = doptics,
        )

    # Builds XICSRT diagnostic
    config = setup._init_config(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        cry_shape = cry_shape,
        niter = niter,
        )

    ########## Monochromatic, point source

    # If rerun the XICSRT calculation
    if pt_run:
        dout = calc.run_mono_pt(
            coll = coll,
            lamb0 = lamb0,
            key_diag = key_diag, key_cam = key_cam,
            config = config,
            cry_shape = cry_shape,
            dpt = dpt,
            run_xicsrt=run_xicsrt, run_tofu=run_tofu,
            dsave = dsave,
            add_velocity = add_velocity, dvel = dvel,
            )

    ########## Monochromatic, volumetric source 

    # If run XICSRT calculation
    if vol_run:
        dout = calc.run_mono_vol(
            coll = coll,
            lamb0 = lamb0,
            key_diag = key_diag, key_cam = key_cam,
            key_mesh = key_mesh, key_lamb = key_lamb, key_emis = key_emis,
            config = config,
            subcam = subcam,
            add_velocity = add_velocity, dvel = dvel,
            # HPC controls
            run_xicsrt=run_xicsrt, run_tofu=run_tofu,
            dHPC = dHPC,
            dvol = dvol,
            dsave = dsave,
            dvos_tf = dvos_tf,
            )


    ########## Multi-energy, volumetrix source  
    if me_run:
        dout = calc.run_multi_vol(
            coll = coll,
            lamb0 = lamb0,
            key_diag = key_diag, key_cam = key_cam,
            key_mesh = key_mesh, key_lamb = key_lamb, key_emis = key_emis,
            config = config,
            subcam = subcam,
            # HPC controls
            run_xicsrt=run_xicsrt, run_tofu=run_tofu,
            dHPC = dHPC,
            dvol = dvol,
            dsave = dsave,
            dvos_tf = dvos_tf,
            )

    ########## Radially-peaked emissivity data
    if rad_run:
        dout = calc.run_rad_emis(
            coll = coll,
            emis_file = emis_file,
            key_diag = key_diag, key_cam = key_cam,
            key_mesh = key_mesh, key_lamb = key_lamb, key_emis = key_emis,
            config = config,
            # HPC controls
            run_xicsrt=run_xicsrt, run_tofu=run_tofu,
            dHPC = dHPC,
            dvol = dvol,
            dsave = dsave,
            dvos_tf = dvos_tf,
            ) 

    ########## Spatial-/spectral-resolution data
    if res_run:
        dout = calc.run_resolution(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            config = config,
            lamb0 = lamb0,
            cry_shape = cry_shape,
            dpt = dpt,
            dres = dres,
            # HPC controls
            run_xicsrt = run_xicsrt,
            run_tofu = run_tofu,
            dsave = dsave,
            )

    # Output
    return dout, coll


###################################################
#
#           Extra
#
###################################################

# Saves XICSRT results
def _save(
    dout=None,
    case=None,
    lamb0 = None, # [AA]
    key_diag = None,
    dsave = None,
    dHPC=None
    ):

    name = os.path.join(
        dsave['path'],
        dsave['name']
        )
    if lamb0 is not None:
        name += '_lamb%1.5fAA'%(lamb0)
    if key_diag is not None:
        name += '_'+key_diag
    if case in ['mv', 'me', 'emis']:
        name += '_job%i'%(dHPC['job_num'])
    if case in ['emis']:
        name += '_ygroup%i'%(dHPC['lamb_num'])
    name += '.npz'

    # Saves XICSRT results
    np.savez(name,dout)

# Organizes HPC output for mono-energetic, volumetric case
def _get_mv_results(
    folder = None,
    folder_xi = None,
    folder_tf = None,
    name_xi = None,
    name_tf = None,
    case = 'mv',
    save_all = True,
    ):
    from cmp_tofu_xicsrt.utils import _conv as cv

    # Init
    dout = {}

    if name_xi is not None:
        # Gets list of output files
        xi_fils = [f for f in os.listdir(folder+'/'+folder_xi) if f.startswith(name_xi)] 

        # Init
        dxi = {}
        optics = ['source', 'crystal', 'detector']

        # Loads XICSRT data
        for xi in xi_fils:
            print(xi)
            key = xi.split('job')[-1].split('.')[0]
            dxi[key] = np.load(
                os.path.join(folder,folder_xi,xi),
                allow_pickle=True
                )['arr_0'][()]['XICSRT']

            if case == 'mv':
                for op in optics:
                    for kk in dxi[key][op].keys():
                        dxi[key][op][kk] = cv._xicsrt2tofu(
                            data=dxi[key][op][kk]
                            )

            # Adds together signal data
            if 'signal' not in dxi.keys():
                dxi['signal'] = np.zeros(dxi[key]['signal'].shape)

                if 'voxels' in dxi[key].keys():  ############### Temporary version control fix
                    dxi['voxels'] = copy.deepcopy(dxi[key]['voxels'])
                dxi['extent'] = copy.deepcopy(dxi[key]['extent'])
                dxi['aspect'] = copy.deepcopy(dxi[key]['aspect'])
                dxi['cents_cm'] = copy.deepcopy(dxi[key]['cents_cm'])
                dxi['npix'] = copy.deepcopy(dxi[key]['npix'])
                if case == 'me':
                    dxi['lambda_AA'] = copy.deepcopy(dxi[key]['lambda_AA'])
                    dxi['dispersion'] = np.zeros(dxi[key]['signal'].shape +dxi[key]['lambda_AA'].shape)
            dxi['signal'] += dxi[key]['signal']
            if case == 'me':
                dxi['dispersion'] += dxi[key]['dispersion']

            if not save_all:
                del dxi[key]

        # Saves
        dout['XICSRT'] = dxi

    # Loads ToFu data
    if name_tf is not None:
        tf_fils = [f for f in os.listdir(folder+'/'+folder_tf) if f.startswith(name_tf)] 
        print(tf_fils)
        dtf = np.load(
            os.path.join(folder,folder_tf,tf_fils[0]), 
            allow_pickle=True
            )['arr_0'][()]['ToFu']

        # Saves
        dout['ToFu'] = dtf

    # Output
    return dout