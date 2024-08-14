'''

Function to prepare configuration data for XICSRT simulations

cjperks
Aug 5, 2024

'''

# Modules
import numpy as np
import copy

import cmp_tofu_xicsrt.utils as utils

__all__ = [
    '_init_config',
    '_build_source'
    ]

###################################################
#
#           Building XICSRT diag from ToFu
#
###################################################

# Function to build XICSRT geometry input from ToFu Collection object
def _init_config(
    coll = None,
    key_diag = None,
    key_cam = None,
    cry_shape = None,
    niter = None,
    ):

    # -----------------
    # Numerical controls
    # -----------------

    # 1. Init
    config = dict()

    # 2. Ray-tracing numerics
    config['general'] = {}
    config['general']['number_of_iter'] = niter
    config['general']['save_images'] = False

    # -----------------
    # extract keys from ToFu coll object
    # -----------------

    # extract dict of optics
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']

    # extract keys of optical objects
    key_cryst = doptics[key_cam]['optics'][0]
    lkey_ap = doptics[key_cam]['optics'][1:]

    # -----------------------------------------------------------
    # Builds apertures
    # -----------------------------------------------------------

    # NOTE: Seems like ToFu lists optics from closest to camera towards source,
    # but XICSRT assumes you build optics from source toward detector

    # Init optics
    config['optics'] = {}

    # Create XICSRT apertures
    for kap in np.flip(lkey_ap):
        # ToFu data
        config['optics'][kap] = {}
        dgeom_ap = coll.dobj['aperture'][kap]['dgeom']

        # Defines object
        config['optics'][kap]['class_name'] = 'XicsrtOpticAperture'

        # Object origin
        config['optics'][kap]['origin'] = utils._tofu2xicsrt(
            data = dgeom_ap['cent'] # [m]
            )

        # Object orientation
        config['optics'][kap]['zaxis'] = utils._tofu2xicsrt(
            data = dgeom_ap['nin'] # Normal
            )
        config['optics'][kap]['xaxis'] = utils._tofu2xicsrt(
            data = dgeom_ap['e0']  # Horizontal
            )

        # Assures y-axis is up
        if np.sum(np.cross(
            config['optics'][kap]['zaxis'], config['optics'][kap]['xaxis']
            )) < 0:
            config['optics'][kap]['xaxis'] *= -1
        
        # Object full extent
        config['optics'][kap]['xsize'] = 2*dgeom_ap['extenthalf'][0] # [m], Horizontal
        config['optics'][kap]['ysize'] = 2*dgeom_ap['extenthalf'][1] # [m], Vertical


    # -----------------------------------------------------------
    # Builds crystal
    # -----------------------------------------------------------

    # ToFu data
    dgeom_cryst = coll.dobj['crystal'][key_cryst]['dgeom']
    dmat_cryst = coll.dobj['crystal'][key_cryst]['dmat']

    # Init
    config['optics']['crystal'] = {}
    config['optics']['crystal']['check_size'] = True # Not exactly sure what this does!!!

    # Object origin
    config['optics']['crystal']['origin'] = utils._tofu2xicsrt(
        data = dgeom_cryst['cent'] # [m]
        )
    
    # Object orientation
    config['optics']['crystal']['zaxis'] = utils._tofu2xicsrt(
        data = dgeom_cryst['nin'] # Normal
        )
    config['optics']['crystal']['xaxis'] = utils._tofu2xicsrt(
        data = dgeom_cryst['e0']  # Horizontal
        )

    # Assures y-axis is up
    if np.sum(np.cross(
        config['optics']['crystal']['zaxis'], config['optics']['crystal']['xaxis']
        )) < 0:
            config['optics']['crystal']['xaxis'] *= -1
    
    # 2d spacing
    config['optics']['crystal']['crystal_spacing'] = dmat_cryst['d_hkl']*1e10 # [AA]

    # Spherical crystal controls
    if cry_shape == 'Spherical':
        # Object type
        config['optics']['crystal']['class_name'] = 'XicsrtOpticSphericalCrystal'

        # Object size
        config['optics']['crystal']['xsize'] = (
            2*dgeom_cryst['extenthalf'][0]
            *dgeom_cryst['curve_r'][0]
            ) # [m], Horizontal
        config['optics']['crystal']['ysize'] = (
            2*dgeom_cryst['extenthalf'][1]
            *dgeom_cryst['curve_r'][1]
            ) # [m], Vertical

        # Object radius
        config['optics']['crystal']['radius'] = dgeom_cryst['curve_r'][0] # [m]

    elif cry_shape == 'Cylindrical':
        # Object type
        config['optics']['crystal']['class_name'] = 'XicsrtOpticCylindricalCrystal'

        # Object size
        config['optics']['crystal']['xsize'] = (
            2*dgeom_cryst['extenthalf'][0]
            *dgeom_cryst['curve_r'][0]
            ) # [m], Horizontal
        config['optics']['crystal']['ysize'] = (
            2*dgeom_cryst['extenthalf'][1]
            #*dgeom_cryst['curve_r'][1]
            ) # [m], Vertical

        # Object radius
        config['optics']['crystal']['radius'] = dgeom_cryst['curve_r'][0] # [m]

    # Error check
    else:
        print('NOT IMPLEMENTED YET!!!')

    # Rocking curve
    ############ NOTE: TO BE GENERALIZED!!!
    '''
    config['optics']['crystal']['rocking_type'] = 'file'
    config['optics']['crystal']['rocking_filetype'] = 'xop'
    config['optics']['crystal']['rocking_file'] = os.path.join(
        '/home/cjperks',
        'atomic_world/run_XICSRT/scripts/',
        'SPARC/rc_161',
        'rc_'+"{:1.5F}".format(1.61)+'.dat'
        )

    config['optics']['crystal']['check_bragg'] = True
    config['optics']['crystal']['rocking_type'] = 'step'
    config['optics']['crystal']['rocking_fwhm'] = 4.515097569188164e-06*2
    '''

    config['optics']['crystal']['rocking_type'] = 'tofu'
    config['optics']['crystal']['rocking_material'] = 'Germanium'
    config['optics']['crystal']['rocking_miller'] = np.r_[2., 0., 2.,]



    # -----------------------------------------------------------
    # Builds camera
    # -----------------------------------------------------------

    # create xicsrt camera
    dgeom_cam = coll.dobj['camera'][key_cam]['dgeom']

    # Init
    config['optics']['detector'] = {}
    config['optics']['detector']['class_name'] = 'XicsrtOpticDetector'

    # Object origin
    config['optics']['detector']['origin'] = utils._tofu2xicsrt(
        data = dgeom_cam['cent'] # [m]
        )
    
    # Object orientation
    config['optics']['detector']['zaxis']  = utils._tofu2xicsrt(
        data = dgeom_cam['nin'] # Normal
        )
    config['optics']['detector']['xaxis']  = -1*utils._tofu2xicsrt(
        data = dgeom_cam['e0']  # Horizontal
        )

    # Assures y-axis is up
    if np.sum(np.cross(
        config['optics']['detector']['zaxis'], config['optics']['detector']['xaxis']
        )) < 0:
            config['optics']['detector']['xaxis'] *= -1
    
    # Object size
    config['optics']['detector']['xsize']  = (
        dgeom_cam['shape'][0]
        *2*dgeom_cam['extenthalf'][0]
        ) # [m], Horizontal
    config['optics']['detector']['ysize']  = (
        dgeom_cam['shape'][1]
        *2*dgeom_cam['extenthalf'][1]
        ) # [m], Vertical

    # Output geometry
    return config



# Builds XICSRT source
def _build_source(
    config =None,
    lamb0 = None,           # [AA]
    box_cent = None,        # [m], dim(x,y,z) ToFu basis
    box_vect = None,        # [vector], list[norm,vert,binorm], ToFu basis
    box_dl = None,          # [m], dim(norm, vert, binorm)
    omega_norm = None,      # [vector], dim(x,y,z) ToFu basis
    omega_vert = None,      # [vector], dim(x,y,z) ToFu basis
    omega_binorm = None,    # [vector], dim(x,y,z) ToFu basis
    omega_dl = None,        # [rad], list[vert, birnorm]
    dHPC = None,
    ):

    # Init
    config_out = copy.deepcopy(config)

    # Init
    config_out['sources'] = {}
    config_out['sources']['source'] = {} # Only one source is supported at this time

    # Source origin
    config_out['sources']['source']['origin'] = utils._tofu2xicsrt(
        data = box_cent
        )

    # Source orientation
    config_out['sources']['source']['zaxis'] = utils._tofu2xicsrt(
        data = -1*box_vect[0]
        )

    # Source size
    config_out['sources']['source']['zsize'] = box_dl[0]
    config_out['sources']['source']['ysize'] = box_dl[2]
    config_out['sources']['source']['xsize'] = box_dl[1]

    # Source type
    #config_out['sources']['source']['class_name'] = 'XicsrtSourceDirected'
    config_out['sources']['source']['class_name'] = 'XicsrtSourceFocused'

    # Number of rays to emit
    config_out['sources']['source']['intensity'] = dHPC['num_rays']

    # Wavelength distirbution
    config_out['sources']['source']['wavelength_dist'] = 'monochrome'
    config_out['sources']['source']['wavelength'] = lamb0 # [AA]

    # Angular distribution
    config_out['sources']['source']['angular_dist'] = 'isotropic_xy' # Horizontal & Vertical extent
    config_out['sources']['source']['spread'] = [
        np.max([abs(omega_dl[0,0]), abs(omega_dl[0,1])])*1.1,
        np.max([abs(omega_dl[1,0]), abs(omega_dl[1,1])])*1.1
        ] # [rad], (binorm,vert), half-angles

    # Emission direction
    #config_out['sources']['source']['direction'] = utils._tofu2xicsrt(
    #    data = omega_norm
    #    )
    config_out['sources']['source']['target'] = config['optics']['ap']['origin']

    # Ouput
    return config_out

