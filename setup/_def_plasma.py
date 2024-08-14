'''

Default values to model plasma

cjperks
Aug 5, 2024

'''

# Modules
import numpy as np

__all__ = [
    'get_dplasma'
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
            'R0': 1.85,     # [m]
            'aa': 0.57,     # [m]
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