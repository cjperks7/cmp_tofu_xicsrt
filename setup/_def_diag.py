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
            },
        'XRSHRKr':{ # slit03
            'cent': np.r_[14.2       , -1.61802349,  0.048     ],
            'nin': np.r_[-0.98966998,  0.14336434,  0.        ],
            'e0': np.r_[-0.14336434, -0.98966998, -0.        ],
            'e1': np.r_[ 0., -0.,  1.],
            'outline_x0': np.r_[-5.e-05,  5.e-05,  5.e-05, -5.e-05],
            'outline_x1': np.r_[-0.0125, -0.0125,  0.0125,  0.0125],
            },
        'XRSHRXe':{ # slit23
            'cent': np.r_[14.2       , -1.61802349,  -0.012     ],
            'nin': np.r_[-0.98966998,  0.14336434,  0.        ],
            'e0': np.r_[-0.14336434, -0.98966998, -0.        ],
            'e1': np.r_[ 0., -0.,  1.],
            'outline_x0': np.r_[-5.e-05,  5.e-05,  5.e-05, -5.e-05],
            'outline_x1': np.r_[-0.0125, -0.0125,  0.0125,  0.0125],
            },
        }

    # Output
    return dap[option]

# Crystal geometry defaults
def get_cry_dgeom(option=None):

    # Crystal flattened dimensions
    xsize = 0.034 # [m], horizontal total length
    ysize = 0.012 # [m], vertical total length
    radius = 0.4

    # Crystal center coordinate
    cent = np.r_[ 1.85000000e+01, -2.24092473e+00,  1.80000000e-02] # [m]

    # Crystal axis basis
    nin = np.r_[-0.26820408,  0.96336212,  0.        ] # normal axis
    e0 = np.r_[0.96336212, 0.26820408, 0. ] # horizontal axis
    e1 = np.r_[ 0.,  0., -1.] # vertical axis

    dgeom = {
        'Spherical':{
            'cent': cent,
            'nin': nin,
            'e0': e0,
            'e1': e1,
            'extenthalf': np.r_[xsize/2/radius, ysize/2/radius], # [rad, rad]
            'curve_r': np.r_[radius, radius], # [m]
            },
        'Cylindrical':{
            'cent': cent,
            'nin': nin,
            'e0': e0,
            'e1': e1,
            'extenthalf': np.r_[xsize/2, ysize/2/radius], # [m, rad]
            'curve_r': np.r_[np.inf, radius], # radius of curvature in plane (e0, e1) & nin --> thus cylinder axis (e0,e1)\cross nin
            },
        'XRSHRKr':{
            'cent': np.r_[18.5, -2.24092473, 0.048],
            'nin': np.r_[-0.27492302,  0.96146624,  0.],
            'e0': np.r_[0.96146624, 0.27492302, 0.],
            'e1': e1,
            'extenthalf': np.r_[0.04, 0.014/0.6], # [m, rad]
            'curve_r': np.r_[np.inf, 0.6], # radius of curvature in plane (e0, e1) & nin --> thus cylinder axis (e0,e1)\cross nin
            },
        'XRSHRXe':{
            'cent': np.r_[18.5, -2.24092473, -0.012],
            'nin': np.r_[-0.2715988 ,  0.96241056,  0.],
            'e0': np.r_[0.96241056, 0.2715988 , 0.],
            'e1': e1,
            'extenthalf': np.r_[0.04, 0.014/0.6], # [m, rad]
            'curve_r': np.r_[np.inf, 0.6], # radius of curvature in plane (e0, e1) & nin --> thus cylinder axis (e0,e1)\cross nin
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
            },
        'XRSHRKr':{
            'material': 'Germanium',
            'miller': np.r_[2., 4., 2.],
            'name': 'Ge242',
            'target': {'lamb': 0}
            },
        'XRSHRXe':{
            'material': 'Quartz',
            'miller': np.r_[1., 0., 1.],
            'name': 'Qu101',
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
    elif option == 'e1M3':
        # Camera size
        extenthalf = [77.1e-3/2, 79.65e-3/2]

        # Camera pixel density
        nx0 = 1028
        nx1 = 1062

    pix_width = 2* extenthalf[0]/nx0
    pix_height = 2* extenthalf[1]/nx1

    dcam = {
        'default':{
            'cent': np.r_[ 1.87931835e+01, -2.00350423e+00,  1.80000000e-02],
            'nin': np.r_[-0.76797874, -0.64047533,  0.        ],
            'e0': np.r_[ 0.64047533, -0.76797874, -0.        ],
            'e1': np.r_[ 0., -0.,  1.],
            'outline_x0': 0.5* pix_width * np.r_[-1, 1, 1, -1],
            'outline_x1': 0.5* pix_height * np.r_[-1, -1, 1, 1],
            'cents_x0': extenthalf[0] * np.linspace(-1, 1, nx0),
            'cents_x1': extenthalf[1] * np.linspace(-1, 1, nx1),
            },
        'e1M3':{
            'cent': np.r_[ 1.87949039e+01, -1.99162819e+00,  1.80000000e-02],
            'nin': np.r_[-0.76797874, -0.64047533,  0.        ],
            'e0': np.r_[ 0.64047533, -0.76797874, -0.        ],
            'e1': np.r_[ 0., -0.,  1.],
            'outline_x0': 0.5* pix_width * np.r_[-1, 1, 1, -1],
            'outline_x1': 0.5* pix_height * np.r_[-1, -1, 1, 1],
            'cents_x0': extenthalf[0] * np.linspace(-1, 1, nx0+1)[:-1] + pix_width/2,
            'cents_x1': extenthalf[1] * np.linspace(-1, 1, nx1+1)[:-1] + pix_height/2,
            },
        }

    # Output
    return dcam[option]

# Get subcamera default options
def get_dsubcam(option=None):

    dsub = {
        'Spherical':{
            'dx': -0.167/1e2,       # [m], Move horizontally
            'xhalfsize': 0.045/1e2, # [m]
            'dy': None,             # [m], Move vertically
            'yhalfsize': None,       # [m]
            'method': 'refine'
            },
        'XRSHRKr':{ # lamb0 = 0.945 AA
            'dx': 0.0044,       # [m], Move horizontally
            'xhalfsize': 75e-5/2, # [m]
            'dy': 0.0305,             # [m], Move vertically
            'yhalfsize': 0.01,       # [m]
            'method': 'zoom'
            },
        'XRSHRXe':{ # lamb0 = 2.72 AA
            'dx': 0.00278,       # [m], Move horizontally
            'xhalfsize': 75e-5/2, # [m]
            'dy': -0.0286,             # [m], Move vertically
            'yhalfsize': 0.01,       # [m]
            'method': 'zoom'
            },
        }

    # Output
    return dsub[option]