'''

Default values to model plasma

cjperks
Aug 5, 2024

'''

# Modules
import numpy as np
import os

__all__ = [
    'get_dplasma',
    'get_dvel'
    ]

####################################
#
#           Main
#
####################################

# Default plasma geometry
def get_dplasma(option=None):

    key = 'SPARC'

    # Tokamak parameters !!! Specific to SPARC !!!
    dtok = {
        'SPARC':{
            'Zmax': 9/100,  # [m]
            'Zmin' :-5/100, # [m]
            'R0': 1.85,     # [m], major radius
            'aa': 0.57,     # [m], minor radius
        }
    }
    R0 = dtok[key]['R0']
    aa = dtok[key]['aa']
    Zmax = dtok[key]['Zmax']
    Zmin = dtok[key]['Zmin']

    dplasma = {
        'default':{
            'crop_poly': np.array([
                [R0-aa, R0+aa, R0+aa, R0-aa, R0-aa], 
                [Zmin, Zmin, Zmax, Zmax, Zmin+0.1/100]
                ])
            }
        }
    dout = dplasma[option]

    # Includes Tokamak-specific data
    for kk in dtok[key].keys():
        dout[kk] = dtok[key][kk]

    # Output
    return dout


# Gets default toroidal velocity settings
def get_dvel(option=None):

    # 1D flux-function parameters
    rho = np.linspace(0,1,1001)
    c1 = 2
    c2 = 2
    vtor0 = 100e3 # [m/s]

    # Default parameters
    dvel = {
        'default': {
            'geqdsk': os.path.join(
                '/home/cjperks',
                'tofu_sparc/background_plasma',
                'PRD_plasma/run1',
                'input.geq'
                ),
            'xquant': 'sq. norm. pol. flux',
            'rho': rho,
            'vel': vtor0 * (1-rho**c1)**c2,
            }
        }

    # Output
    return dvel[option]