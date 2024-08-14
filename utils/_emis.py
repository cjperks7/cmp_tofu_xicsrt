'''

Functions to manage local emissivity

cjperks
Aug 5th, 2024

'''

# Modules
import numpy as np

__all__ = [
    '_conv_2normEmis'
    ]

# Converts #rays to normalized emissivity
def _conv_2normEmis(
    voxels=None,
    case=None
    ):
    '''
    If point-source: normalize to 1 ph/s
    If volumetric source: normalize to 1 ph/s/cm^3
    *assuming num_rays= flux from a steady-state plasma over 1s
    '''

    # Init
    dt = 1 # [s]
    src = voxels['num_rays'] / dt # [ph/s]

    # Differential solid angle
    src *= (
        4*np.pi
        /voxels['dOmega_ster']
        ) # [ph/s]

    # Differential volume
    if case in ['mv', 'me']:
        src /= voxels['dvol_cm3'] # [ph/s/cm^3]

    # Output
    return src

