'''

Stores default values for building voxels

'''

# Modules 
import numpy as np

__all__ = [
    'get_dvol'
    ]

#########################################
#
#           Main
#
#########################################

# Defaults for voxel sizes
def get_dvol(
    option = None,
    ):

    # Options
    dvol = {
        'default': {
            'nsteps': {     # Number of steps about central box
                'nn': 124,#62,
                'nz': 3,
                'nb': 2
                },
            'lsteps': {     # Size of steps
                'ln': 0.5/100, # [m]
                'lz': 2/100, # [m]
                'lb': 0.5/100 # [m]
                },
            'plt': False,
            }
        }

    # Output
    return dvol[option]