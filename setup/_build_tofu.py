'''

Function to build diagnostic as a ToFu Collection object

cjperks
Aug 5, 2024

'''

# Modules
import tofu as tf

import numpy as np

import cmp_tofu_xicsrt.setup._def_diag as dd
import cmp_tofu_xicsrt.utils as utils

__all__ = [
    '_init_diag'
    ]

#################################################
#
#               Main
#
#################################################

# Load diagnostic from ToFu
def _init_diag(
    coll = None,
    dplasma = None,
    dcry = None,
    lamb0 = None,
    ):

    # If user is reloading a Collection object
    if isinstance(coll, str):
        coll = tf.data.load(coll)

    # Else, build a new diagnostic
    else:
        coll = _build_diag(
            dcry = dcry,
            lamb0 = lamb0
            )

        # Saves diagnostic
        coll.save(path='/home/cjperks/cmp_tofu_xicsrt/diags')

    # Output
    return coll


#################################################
#
#               Utilities
#
#################################################

# Script to build diagnostic in ToFu
def _build_diag(
    dap = None,
    dcry = None,
    dmat = None,
    dcam = None,
    lamb0 = None,
    ):

    # Init
    coll = tf.data.Collection()

    # Gets default aperture geometry
    if dap is None:
        dap = dd.get_dap(option='default')

    # Adds aperture
    coll.add_aperture(
        key='ap',
        **dap
        )

    # Gets dafault crystal geometry
    if isinstance(dcry, str):
        dcry = dd.get_cry_dgeom(option=dcry)
    if dmat is None:
        dmat = dd.get_cry_dmat(option='default')
    dmat['target']['lamb'] = lamb0*1e-10

    # Adds crystal
    coll.add_crystal(
        key = 'cry',
        dgeom = dcry,
        dmat = dmat,
        )

    # Gets default camera geometry
    if dcam is None:
        dcam = dd.get_dcam(option='default')

    # Adds camera
    coll.add_camera_2d(
        key = 'cam',
        dgeom = dcam 
        )

    # Builds diagnostic
    coll.add_diagnostic(
        key = 'valid',
        doptics = {'cam': ['cry', 'ap']},
        compute = True, # compute LOS
        compute_vos_from_los = True,
        convex = True,
        config = tf.load_config('SPARC-V0')
        )

    # Output
    return coll











