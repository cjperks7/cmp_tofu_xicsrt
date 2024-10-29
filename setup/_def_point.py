'''

Stores default values for building point source

'''

# Modules 
import numpy as np

__all__ = [
    'get_dpt'
    ]

#########################################
#
#           Main
#
#########################################

# Defaults for voxel sizes
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
        }

    # Output
    return dpt[option]