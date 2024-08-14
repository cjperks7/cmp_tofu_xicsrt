'''

XICSRT_utils is a function meant to manage various comparison
analyzes between XICSRT and ToFu

cjperks
Mar 19th, 2024

'''

# Modules
import os, sys
import numpy as np

import cmp_tofu_xicsrt.calc as calc
import cmp_tofu_xicsrt.setup as setup


__all__ = [
    'main',
    '_save'
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
    key_diag = 'valid',
    key_cam = 'cam',
    # XICSRT data
    cry_shape = 'Spherical',
    niter = 5,
    # Monochromatic, point source controls
    pt_run = False,
    dpt = None,     # Resolution controls
    # Monochromatic, volumetric source controls
    vol_run = False,
    dvol = None,    # Resolution controls
    # Multi-energy, volumetric source controls
    me_run = False,
    # HPC controls
    run_xicsrt = True,
    run_tofu = False,
    dHPC = None,
    dsave = None,
    ):

    # Builds ToFu diagnostic
    coll = setup._init_diag(
        coll = coll_tf,
        dcry = cry_shape,
        lamb0 = lamb0
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
            key_diag = key_diag,
            key_cam = key_cam,
            config = config,
            dpt = dpt,
            lamb0 = lamb0,
            run_xicsrt=run_xicsrt,
            run_tofu=run_tofu,
            dsave = dsave,
            )

    ########## Monochromatic, volumetric source 

    # If run XICSRT calculation
    if vol_run:
        dout = calc.run_mono_vol(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            config = config,
            dvol = dvol,
            lamb0 = lamb0,
            # HPC controls
            run_xicsrt=run_xicsrt,
            run_tofu=run_tofu,
            dHPC = dHPC,
            dsave = dsave,
            )


    ########## Multi-energy, volumetrix source  
    if me_run:
        dout = calc.run_multi_vol(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            config = config,
            dvol = dvol,
            lamb0 = lamb0,
            # HPC controls
            run_xicsrt=run_xicsrt,
            run_tofu=run_tofu,
            dHPC = dHPC,
            dsave = dsave
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
    dsave = None,
    dHPC=None
    ):

    name = os.path.join(
        dsave['path'],
        (
            dsave['name'] 
            + '_lamb%1.5fAA'%(lamb0)
            )
        )
    if case in ['mv', 'me']:
        name += '_job%i'%(dHPC['job_num'])
    name += '.npz'

    # Saves XICSRT results
    np.savez(name,dout)
