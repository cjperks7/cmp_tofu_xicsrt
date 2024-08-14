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
        'default': {
            'ToFu':{
                'point': np.r_[1.8, 0.18, 0.0],
                'n0': 2001,#10001,
                'n1': 2001,#10001,
                'plt': False,
                },
            'XICSRT':{
                'intensity': 1e6,
                'dOmega': [0.0005, 0.001] # [rad]
                },
            'plotting':{
                'xind': 44,
                }
            }
        }

    # Output
    return dpt[option]