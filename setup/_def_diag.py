'''

Script to store default values for diagnostic build

cjperks
Aug 5, 2024

'''

# Modules
import numpy as np

__all__ = [
    'get_dap'
    ]

#######################################
#
#           Main
#
#######################################

# Aperture geometry defaults
def get_dap(option=None):

    dap = {
        'default':{
            'cent': np.r_[14.2       , -1.61802349,  0.018     ],
            'nin': np.r_[-0.98966998,  0.14336434,  0.        ],
            'e0': np.r_[-0.14336434, -0.98966998, -0.        ],
            'e1': np.r_[ 0., -0.,  1.],
            'outline_x0': np.r_[-0.0025,  0.0025,  0.0025, -0.0025],
            'outline_x1': np.r_[-0.02, -0.02,  0.02,  0.02],
            }
        }

    # Output
    return dap[option]

# Crystal geometry defaults
def get_cry_dgeom(option=None):

    dgeom = {
        'Spherical':{
            'cent': np.r_[ 1.85000000e+01, -2.24092473e+00,  1.80000000e-02],
            'nin': np.r_[-0.26820408,  0.96336212,  0.        ],
            'e0': np.r_[0.96336212, 0.26820408, 0. ],
            'e1': np.r_[ 0.,  0., -1.],
            'extenthalf': np.r_[0.04256678, 0.01489837],
            'curve_r': np.r_[0.4, 0.4],
            },
        'Cylindrical':{
            'cent': np.r_[ 1.85000000e+01, -2.24092473e+00,  1.80000000e-02],
            'nin': np.r_[-0.26820408,  0.96336212,  0.        ],
            'e0': np.r_[0.96336212, 0.26820408, 0. ],
            'e1': np.r_[ 0.,  0., -1.],
            'extenthalf': np.r_[0.04256678, 0.01489837],
            'curve_r': np.r_[0.4,np.inf],#np.r_[np.inf, 0.4],
            },
        }

    # Output
    return dgeom[option]

# Crystal material defaults
def get_cry_dmat(option=None):

    dmat = {
        'default':{
            'material': 'Germanium',
            'miller': np.r_[2., 0., 2.],
            'name': 'Ge202',
            'target': {'lamb': 0}
            }
        }

    # Output
    return dmat[option]

# Camera geometry defaults
def get_dcam(option=None):

    # Get camera dimension
    if option == 'default':
        # Camera size
        extenthalf = [5.20898437e-03, 3.32226562e-03]

        # Camera pixel density
        nx0 = 128
        nx1 = 64

    pix_width = 2* extenthalf[0]/(nx0-1)
    pix_height = 2* extenthalf[1]/(nx1-1)

    dcam = {
        'default':{
            'cent': [ 1.87931835e+01, -2.00350423e+00,  1.80000000e-02],
            'nin': [-0.76797874, -0.64047533,  0.        ],
            'e0': [ 0.64047533, -0.76797874, -0.        ],
            'e1': [ 0., -0.,  1.],
            'outline_x0': 0.5* pix_width * np.r_[-1, 1, 1, -1],
            'outline_x1': 0.5* pix_height * np.r_[-1, -1, 1, 1],
            'cents_x0': extenthalf[0] * np.linspace(-1, 1, nx0),
            'cents_x1': extenthalf[1] * np.linspace(-1, 1, nx1),
            }
        }

    # Output
    return dcam[option]