'''

Stores default values for building point source

'''

# Modules 
import numpy as np

__all__ = [
    'get_dpt',
    'get_dres',
    ]

#########################################
#
#           Main
#
#########################################

# Defaults for point source resolution
def get_dpt(
    option = None,
    ):

    # Options
    dpt = {
        'Spherical': {
            'ToFu':{
                'point': np.r_[1.8, 0.18, 0.0],
                'n0': 1001,#5001,
                'n1': 1001,#5001,
                'plt': False,
                },
            'XICSRT':{
                'intensity': 1e6,
                'dOmega': [0.001, 0.0001] # [rad]
                },
            'plotting':{
                'xind': 43,
                }
            },
        'Cylindrical': {
            'ToFu':{
                'point': np.r_[1.84347549, 0.15654151, 0.018],
                'n0': 2001,#10001,
                'n1': 2001,#10001,
                'plt': False,
                },
            'XICSRT':{
                'intensity': 1e6,
                'dOmega': [
                    0.001, 
                    0.0002
                    ] # [rad]
                },
            'plotting':{
                'xind': None,
                }
            },
        'XRSHRKr': {
            'ToFu':{
                #'point': np.r_[1.89106408, 0.14276134, 0.01514425],
                'point': np.r_[1.88851308, 0.1547071 , 0.04172155],
                #'point': np.r_[1.88843169, 0.15496111, 0.10381247],
                'n0': 5001,#10001,
                'n1': 5001,#10001,
                'plt': False,
                },
            'XICSRT':{
                'intensity': 2e6,
                'dOmega': [
                    0.0011,     # height 
                    0.000018     # width
                    ] # [rad]
                },
            'plotting':{
                'xind': None,
                }
            },
        }

    # Output
    return dpt[option]

# Defaults for spatial-/spectral-resolution
def get_dres(option=None):

    # Options
    dres = {
        'XRSHRKr':{
            'dz': 7/100, # [m], maximum vertical excursion in one direction
            'nz': 11, # number of vertical steps to make
            'dy': 0.1e-3, # [A], maximum spectral excursion in one direction 
            'ny': 31, # number of spectral steps to make
            },
        'XRSHRXe':{
            'dz': 7/100, # [m], maximum vertical excursion in one direction
            'nz': 11, # number of vertical steps to make
            'dy': 1.3e-3, # [A], maximum spectral excursion in one direction 
            'ny': 11, # number of spectral steps to make
            },
        'XRSLR':{
            'dz': 7/100, # [m], maximum vertical excursion in one direction
            'nz': 11, # number of vertical steps to make
            'dy': 0.7e-3, # [A], maximum spectral excursion in one direction 
            'ny': 11, # number of spectral steps to make
            },
        }

    # Output
    return dres[option]