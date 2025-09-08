'''

Handles VOS computation

cjperks
Sept 9th, 2025

'''

# Modules
import numpy as np

import tofu as tf

__all__ = [
    '_compute_vos_tofu',
    '_compute_signal_tofu'
    ]

#################################################################
#
#               TOFU-specific
#
#################################################################

# Computes VOS 
def _compute_vos_tofu(
    # Diag
    coll = None,
    key_diag = None,
    key_mesh = None, key_lamb = None,
    # Controls
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

    # Runs calculation
    dvos, dref = coll.compute_diagnostic_vos(
        key_diag = key_diag,
        key_mesh = key_mesh,
        res_RZ = dvos_tf['res_RZ'],         # 0.005 would be better
        res_phi = dvos_tf['res_phi'],        # 0.0002 would be better
        lamb = coll.ddata['%s_k'%(key_lamb)]['data'],
        n0 = dvos_tf['n0'],
        n1 = dvos_tf['n1'],
        visibility = False,
        config = tf.load_config('SPARC-V0'),
        return_vector = False,
        #keep3d=False,
        store = True,
        )

    # Saves VOS matrix
    if dvos_tf['save']:
        coll.save(path = dvos_tf['path'])

    # Output
    return coll

# Computes diagnostic signal
def _compute_signal_tofu(
    coll = None,
    key_diag = None, key_cam = None,
    key_emis = None,
    ):

    # Computes signal with emissivity
    dsig = coll.compute_diagnostic_signal(
        key = 'flux_vos_'+key_diag,
        key_diag = key_diag,
        key_cam = [key_cam],
        key_integrand = key_emis,
        key_ref_spectro = None,
        method = 'vos',
        res = None,
        mode = None,
        groupby = None,
        val_init = None,
        ref_com = None,
        brightness = False,
        spectral_binning = False,
        dvos = None,
        verb = False,
        timing = False,
        store = True,
        returnas = dict,
        )

    # Output
    return dsig