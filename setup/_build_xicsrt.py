'''

Function to prepare configuration data for XICSRT simulations

cjperks
Aug 5, 2024

'''

# Modules
import numpy as np
import copy
import scipy.constants as cnt

import cmp_tofu_xicsrt.utils as utils
import cmp_tofu_xicsrt.setup._def_plasma as dp

__all__ = [
    '_init_config',
    '_build_source',
    '_add_velocity',
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
    split = True,
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
    
    # 2d spacing
    config['optics']['crystal']['crystal_spacing'] = dmat_cryst['d_hkl']*1e10 # [AA]

    # Spherical crystal controls
    if cry_shape == 'Spherical':
        # Object orientation
        config['optics']['crystal']['zaxis'] = utils._tofu2xicsrt(
            data = dgeom_cryst['nin'] # Normal
            )
        config['optics']['crystal']['xaxis'] = utils._tofu2xicsrt(
            data = dgeom_cryst['e0']  # Horizontal
            )

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
        
        # Normal orientation
        config['optics']['crystal']['zaxis'] = utils._tofu2xicsrt(
            data = dgeom_cryst['nin'] # Normal
            )

        # If the crystal is not curved in the dispersive direction
        if np.isinf(dgeom_cryst['curve_r'][0]):
            # Cylinder orientation
            config['optics']['crystal']['xaxis'] = utils._tofu2xicsrt(
                data = dgeom_cryst['e0']  # NOTE: XICSRT assumes "xaxis" is the cylinder axis (curved about)
                )

            # Object size
            config['optics']['crystal']['xsize'] = (
                2*dgeom_cryst['extenthalf'][0]
                #*dgeom_cryst['curve_r'][0]
                ) # [m], Horizontal size
            config['optics']['crystal']['ysize'] = (
                2*dgeom_cryst['extenthalf'][1]
                *dgeom_cryst['curve_r'][1]
                ) # [m], Vertical size

            # Object radius
            config['optics']['crystal']['radius'] = dgeom_cryst['curve_r'][1] # [m]

        # If the crystal is not curved in the imaging direction
        elif np.isinf(dgeom_cryst['curve_r'][1]):
            # Cylinder orientation
            config['optics']['crystal']['xaxis'] = utils._tofu2xicsrt(
                data = dgeom_cryst['e1']  # NOTE: XICSRT assumes "xaxis" is the cylinder axis (curved about)
                )

            # Object size
            config['optics']['crystal']['xsize'] = (
                2*dgeom_cryst['extenthalf'][1]
                #*dgeom_cryst['curve_r'][0]
                ) # [m], Horizontal size
            config['optics']['crystal']['ysize'] = (
                2*dgeom_cryst['extenthalf'][0]
                *dgeom_cryst['curve_r'][0]
                ) # [m], Vertical size

            # Object radius
            config['optics']['crystal']['radius'] = dgeom_cryst['curve_r'][0] # [m]

        # Error check
        else:
            print('ERROR IN CRYSTAL RADIUS!!!')

    # Error check
    else:
        print('NOT IMPLEMENTED YET!!!')

    # Assures y-axis is up
    if np.sum(np.cross(
        config['optics']['crystal']['zaxis'], config['optics']['crystal']['xaxis']
        )) < 0:
            config['optics']['crystal']['xaxis'] *= -1

    # Rocking curve
    config['optics']['crystal']['rocking_type'] = 'tofu'
    config['optics']['crystal']['rocking_material'] = dmat_cryst['material']
    config['optics']['crystal']['rocking_miller'] = dmat_cryst['miller']

    # -----------------------------------------------------------
    # Builds camera
    # -----------------------------------------------------------

    if split:
        kk = key_cam.split('XRS')[0][:-1]
    else:
        kk = key_cam

    # create xicsrt camera
    dgeom_cam = coll.dobj['camera'][kk]['dgeom']

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
    key_ap = None,
    dHPC = None,
    add_velocity = False,
    dvel = None,
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
    config_out['sources']['source']['target'] = config['optics'][key_ap]['origin']

    # If adding Doppler shift
    if add_velocity:
        config_out['sources']['source']['velocity'] = utils._tofu2xicsrt(
            data = _add_velocity(
                dvel = dvel,
                box_cent = box_cent
                )
            ) # [m/s], clockwise toroidal flow

    # Ouput
    return config_out

###################################################
#
#           Utilities
#
###################################################

# Local toroidal velocity
def _add_velocity(
    dvel = None,
    box_cent = None
    ):
    '''
    dvel is a dictionary containing data about the velocity profile is the form:
    \vec(v)(R,Z) = \omega(\Psi)*R*\hat(\phi) + u(\Psi)*\vect(B) [m/s]
        where,
        \omega and u are flux-functions
        \omega [rad/s] is the solid body rotation frequency
        R [m] is the local major radius
        \hat(\phi) is the toroidal direction
        u [m/s/T] is the poloidal flow
        \vec(B) [T] is the local B-field vector

    The expected format of dvel is
        1) if dvel['option'] == 'fixed'
            Constant toroidal rotation
        2) if dvel['option'] == 'radial'
            Radial profile for toroidal rotation

    TO DO:
        1) Add helicity part

    '''

    # Init
    box_R = np.sqrt(box_cent[0]**2+box_cent[1]**2)
    box_Z = box_cent[2]
    alp = np.arctan(box_cent[1]/box_cent[0])

    # If a constant rotation frequency
    if dvel['option'] == 'fixed':
        # Calculates magnitude of local linear velocity
        box_vel = dvel['omega']*box_R # [m/s]

    # Test case with fixed linear velocity
    elif dvel['option'] == 'test':
        box_vel = dvel['velocity'] # [m/s]

    # If a radial profile for the rotation frequency
    elif dvel['option'] == 'radial':

        from omfit_classes import omfit_eqdsk
        from scipy.interpolate import LinearNDInterpolator, interp1d

        # Gets defaults
        if dvel is None:
            dvel = dplasma = dp.get_dvel(option='default')

        # Loads geqdsk
        geq = omfit_eqdsk.OMFITgeqdsk(dvel['geqdsk'])

        # Gets equilibrium data
        if dvel['xquant'] == 'sq. norm. pol. flux':
            rho_2d = np.sqrt(geq['AuxQuantities']['PSIRZ_NORM']).T # dim(nR,nZ)
        R_1d = geq['AuxQuantities']['R'] # dim(nR,)
        Z_1d = geq['AuxQuantities']['Z'] # dim(nZ,)
        Z_2d, R_2d = np.meshgrid(Z_1d, R_1d) # dim(nR, nZ)

        # Interpolates onto finer mesh on desired precision
        precR = 1e-3 # [m]
        precZ = 1e-3 # [m]

        nR = int(np.ceil((max(R_1d)-min(R_1d))/precR))
        nZ = int(np.ceil((max(Z_1d)-min(Z_1d))/precZ))
        R_fine_1d = np.linspace(min(R_1d), max(R_1d), nR)
        Z_fine_1d = np.linspace(min(Z_1d), max(Z_1d), nZ)
        Z_fine_2d, R_fine_2d = np.meshgrid(Z_fine_1d, R_fine_1d) # dim(nR, nZ)

        rho_fine_2d = LinearNDInterpolator(
            (R_2d.flatten(), Z_2d.flatten()),
            rho_2d.flatten()
            )(
                (R_fine_2d.flatten(), Z_fine_2d.flatten())
                ).reshape(R_fine_2d.shape)

        # Flux-function toroidal velocity
        rho_1d = dvel['rho'] # dim(nrho,)
        omega_1d = dvel['omega'] # [m/s], dim(nrho,)

        # Finds the flux-surface of the box center
        indR = np.argmin(abs(R_fine_1d-box_R))
        indZ = np.argmin(abs(Z_fine_1d-box_Z))
        box_rho = rho_fine_2d[indR,indZ]

        # Finds the velocity at this flux surface
        box_omega = inter1d(
            rho_1d, omega_1d,
            bounds_error=False,
            fill_value = 0.0
            )(box_rho) # [rad/s]
        box_vel = box_omega*box_R # [m/s]

    # Converts to a clockwise toroidal flow
    return np.r_[
        box_vel * np.sin(alp),
        box_vel * -1* np.cos(alp),
        0.0
        ]