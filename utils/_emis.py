'''

Functions to manage local emissivity

cjperks
Aug 5th, 2024

'''

# Modules
import numpy as np
from scipy.interpolate import griddata
import tofu as tf

import cmp_tofu_xicsrt.utils as utils

__all__ = [
    '_conv_2normEmis',
    '_calc_signal',
    '_build_gaussian',
    '_prep_emis_tofu',
    '_prep_emis_xicsrt',
    ]

###################################################
#
#          Prepares emissivity output data
#
###################################################

# Converts #rays to normalized emissivity
def _conv_2normEmis(
    voxels=None,
    case=None
    ):
    '''
    If point-source: normalize to 1 ph/s
    If volumetric source: normalize to 1 ph/s/cm^3
    *assuming num_rays= flux from a steady-state plasma over 1s
    '''

    # Init
    dt = 1 # [s]
    src = voxels['num_rays'] / dt # [ph/s]

    # Differential solid angle
    src *= (
        4*np.pi
        /voxels['dOmega_ster']
        ) # [ph/s]

    # Differential volume
    if case in ['mv', 'me']:
        src /= voxels['dvol_cm3'] # [ph/s/cm^3]

    # Output
    return src


# Calculates signal from VOS
def _calc_signal(
    dout = None,
    config = None,
    case = None,
    demis = None,
    box_cent = None,
    det_origin = None,
    emis_val = 1,
    dlamb = 1,
    ilamb = None,
    nlamb = None,
    nx = 1028,
    ny = 1062,
    ):

    # If local emissivity data given
    if demis is not None:
        # Init
        box_R = np.sqrt(box_cent[0]**2 + box_cent[1]**2)
        box_Z = box_cent[2]

        # Interpolates local value of emissivity
        nR, nZ = demis['mesh_R'].size, demis['mesh_Z'].size
        RR_knots = np.repeat(demis['mesh_R'][:, None], nZ, axis=1) # dim(mesh_R, mesh_Z)
        ZZ_knots = np.repeat(demis['mesh_Z'][None, :], nR, axis=0) # dim(mesh_R, mesh_Z)

        emis_val = griddata(
            (RR_knots.flatten(), ZZ_knots.flatten()),
            demis['emis'].flatten(),
            (box_R, box_Z),
            method='linear') # dim(scalar); [ph/s/cm3/AA]

        # Error check
        if np.isnan(emis_val):
            emis_val = 0.0

        dlamb = demis['dlamb'] # dim(scalar); [AA]
        nlamb = demis['nlamb']
        ilamb = demis['ilamb']

    # Calculates histogram
    dhist = utils._calc_det_hist(
        rays = det_origin,
        config = config,
        nx = nx,
        ny = ny
        ) # dim(hor_pix, vert_pix)
    # wavelength-integration (via left-hand Riemann sum)
    if 'signal' not in dout.keys():
        dout['signal'] = np.zeros(dhist['counts'].shape)
    dout['signal'] += (
        dhist['counts'] 
        / _conv_2normEmis(voxels=dout['voxels'], case=case)
        * emis_val * dlamb
        ) # dim(nx,ny), [photons/bin^2]

    # Includes dispersion if multi-energy case
    if case == 'me':
        if 'dispersion' not in dout.keys():
            dout['dispersion'] = np.zeros(dhist['counts'].shape+ (nlamb,))
        dout['dispersion'][:,:,ilamb] += dhist['counts'] 

    # Output
    return dout

###################################################
#
#          Prepares emissivity input data
#
###################################################

# Formats emissivity data for convenient use in XICSRT
def _prep_emis_xicsrt(
    coll = None,
    key_diag = None,
    key_mesh = None,
    ilamb = None,
    dlamb = None,
    nlamb = None,
    ):

    # Output
    return {
        'emis': (
            coll.ddata['emis_'+key_diag]['data'][:,:,ilamb]
            * (4*np.pi) *1e-6 * 1e-10 # Converts from ph/s/m3/sr/m -> ph/s/cm3/AA
            ), # dim(mesh_R, mesh_Z); [ph/s/cm3/AA]
        'mesh_R': coll.ddata['%s_k0'%(key_mesh)]['data'], # dim(mesh_R); [m]
        'mesh_Z': coll.ddata['%s_k1'%(key_mesh)]['data'], # dim(mesh_Z); [m]
        'dlamb': dlamb, # dim(scalar); [AA]
        'nlamb': nlamb,
        'ilamb': ilamb,
        }

# Imoort the emissivity data into Collection object 
def _prep_emis_tofu(
    coll = None,
    case = None,
    key_diag = None, #key_cam = None,
    key_mesh = None, key_lamb = None, key_emis = None,
    emis_file = None,
    lamb0 = None,
    conf = None,
    # R,Z discretization
    R_knots = None,
    Z_knots = None,
    # wavelength discretization
    nlamb = 500
    ):

    # Loads data
    if emis_file is not None:
        demis = np.load(
            emis_file,
            allow_pickle=True
            )['arr_0'][()]
        R_knots = emis['plasma']['RR']['data'] # [m], dim(R,)
        Z_knots = emis['plasma']['ZZ']['data'] # [m], dim(Z,)
        rhop_RZ=np.sqrt(emis['plasma']['PSIN_RZ']['data']).T
    else:
        R_knots = None
        Z_knots = None
        rhop_RZ = None
    
    # Adds (R,Z) mesh data to Collection object
    coll = utils._build_RZ_mesh_tofu(
        coll = coll,
        key_mesh = key_mesh,
        conf = conf,
        case = case,
        dplasma = None,
        # Data for flux-function map
        R_knots = R_knots,
        Z_knots = Z_knots,
        rhop_RZ = rhop_RZ,
        )

    # If running a monochromatic, volumetric source
    if case == 'mv':
        coll = _build_emis_box(
            coll = coll,
            key_mesh = key_mesh, key_lamb = key_lamb, key_emis = key_emis,
            lamb0 = lamb0,
            )

    # If running a Gaussian in energy, constant in space source
    elif case == 'me':
        coll = _build_emis_gaussian(
            coll = coll,
            key_mesh = key_mesh, key_lamb = key_lamb, key_emis = key_emis,
            lamb0 = lamb0,
            )

    # If running an emissivity map from tofu_sparc
    elif case == 'rad_emis':
        coll = _build_emis_tfs(
            coll = coll,
            key_diag = key_diag,
            key_mesh = key_mesh, key_lamb = key_lamb, key_emis = key_emis,
            nlamb = nlamb,
            demis = demis,
            R_knots = R_knots, Z_knots = Z_knots,
            )

    # Output
    return coll

###################################################
#
#           Emissivity types
#
###################################################

# If emissivity data is generated using tofu_sparc
def _build_emis_tfs(
    coll = None,
    key_diag = None,
    key_mesh = None, key_lamb = None, key_emis = None,
    nlamb = None,
    # tofu_sparc data
    demis = None,
    R_knots = None,
    Z_knots = None,
    ):

    # Init
    diag_com = key_diag.split('_')[0] # Common name

    ########### ----- Add wavelength data ------ ############

    # Number of points to skip on fine wavelength mesh
    fact = int(demis[diag_com]['emis']['data'].shape[-1]/nlamb)

    # Ensures wavelength data is monotonically increasing
    lamb_vec = demis[diag_com]['lambda']['data'][::fact]
    flip = False
    if np.mean(lamb_vec[1:]-lamb_vec[:-1])<0:
        lamb_vec = np.flip(lamb_vec)
        flip = True

    # Adds wavelength mesh to Collection object
    coll = utils._build_lamb_mesh_tofu(
        coll = coll,
        key_lamb = key_lamb,
        lamb_vec = lamb_vec,    # [AA]
        )

    ########### ----- Add emissivity data ------ ############

    # Interpolates from transport grid onto regular grid
    nR, nZ = R_knots.size, Z_knots.size
    RR_knots = np.repeat(R_knots[:, None], nZ, axis=1) # dim(mesh_R, mesh_Z)
    ZZ_knots = np.repeat(Z_knots[None, :], nR, axis=0) # dim(mesh_R, mesh_Z)
    emis_RZ = _interp_emis(
        emis=demis[diag_com]['emis']['data'], # dim(fm_rhop, fm_theta, fm_nlamb)
        eta=demis[diag_com]['eta']['data'], # dim(fm_nlamb)
        nlamb=nlamb,
        fact=fact,
        RR_knots=RR_knots, # dim(mesh_R, mesh_Z)
        ZZ_knots=ZZ_knots, # dim(mesh_R, mesh_Z)
        RR_asym=demis['plasma']['rhop_contours']['R']['data'], # dim(fm_rhop, fm_theta),
        ZZ_asym=demis['plasma']['rhop_contours']['Z']['data'], # dim(fm_rhop, fm_theta)
        ) # dim(mesh_R, mesh_Z, nlamb)

    if flip:
        emis_RZ = np.flip(emis_RZ, axis=-1)

    # Adds emissivity mesh data to Collection object
    coll = utils._build_emis_mesh_tofu(
        coll = coll,
        key_mesh = key_mesh,
        key_lamb = key_lamb,
        key_emis = key_emis,
        # Data
        emis_RZ = emis_RZ, # dim(R,Z,lambda); [ph/s/m3/sr/m]
        #emis_1d = None, # dim(lambda,); [ph/s/m3/sr/m]
        )

    # Output
    return coll

# Box function in energy, constant in space
def _build_emis_box(
    coll = None,
    key_mesh = None, key_lamb = None, key_emis = None,
    lamb0 = None,
    ):

    # Gets box shape
    dtol = 1e-6         # [AA]
    lamb_vec = np.r_[
        -dtol/2,
        dtol/2
        ] + lamb0 # [AA]

    # Prepares Gaussian emissivity at 1 ph/s/cm3
    emis = np.ones_like(lamb_vec)*(
        1e6 # [ph/s/m3]
        /dtol*1e10 # [1/m]
        / (4*np.pi) # [1/sr]
        ) # [ph/s/m3/sr/m], dim(nlamb,)

    # Assumes spatially homogeneous
    nR, nZ = coll.dobj['mesh'][key_mesh]['shape_k']
    emis_RZ = np.repeat(
        np.repeat(emis[None, None, :], nR, axis=0),
        nZ,
        axis=1,
        )

    # Adds wavelength mesh to Collection object
    coll = utils._build_lamb_mesh_tofu(
        coll = coll,
        key_lamb = key_lamb,
        lamb_vec = lamb_vec,    # [AA]
        )

    # Adds emissivity mesh data to Collection object
    coll = utils._build_emis_mesh_tofu(
        coll = coll,
        key_mesh = key_mesh,
        key_lamb = key_lamb,
        key_emis = key_emis,
        # Data
        emis_RZ = emis_RZ, # dim(R,Z,lambda); [ph/s/m3/sr/m]
        #emis_1d = None, # dim(lambda,); [ph/s/m3/sr/m]
        )

    # Output
    return coll

# Gaussian in energy, constant in space
def _build_emis_gaussian(
    coll = None,
    key_mesh = None, key_lamb = None, key_emis = None,
    lamb0 = None,
    ):

    # Gets Gaussian
    lamb_vec, fE = _build_gaussian(lamb0=lamb0) # [AA], [1/AA], dim(nlamb,)

    # Prepares Gaussian emissivity at 1 ph/s/cm3
    emis = (
        1e6 # [ph/s/m3]
        *fE*1e10 # [1/m]
        / (4*np.pi) # [1/sr]
        ) # [ph/s/m3/sr/m], dim(nlamb,)

    # Assumes spatially homogeneous
    nR, nZ = coll.dobj['mesh'][key_mesh]['shape_k']
    emis_RZ = np.repeat(
        np.repeat(emis[None, None, :], nR, axis=0),
        nZ,
        axis=1,
        )

    # Adds wavelength mesh to Collection object
    coll = utils._build_lamb_mesh_tofu(
        coll = coll,
        key_lamb = key_lamb,
        lamb_vec = lamb_vec,    # [AA]
        )

    # Adds emissivity mesh data to Collection object
    coll = utils._build_emis_mesh_tofu(
        coll = coll,
        key_mesh = key_mesh,
        key_lamb = key_lamb,
        key_emis = key_emis,
        # Data
        emis_RZ = emis_RZ, # dim(R,Z,lambda); [ph/s/m3/sr/m]
        #emis_1d = None, # dim(lambda,); [ph/s/m3/sr/m]
        )

    # Output
    return coll

###################################################
#
#           Utilities
#
###################################################

# Builds wavelength mesh for a simple Gaussian
def _build_gaussian(
    lamb0 = None, # [AA], centroid
    slamb = 5e-4, # [AA], std
    nlamb = 61, # num grid points
    xlamb = 3.2, # num of std's
    ):

    # Wavelength mesh, [AA]
    lamb = np.linspace(
        lamb0 - xlamb*slamb,
        lamb0 + xlamb*slamb,
        nlamb
        )

    # Normalized distribution, [1/AA]
    fE = (
        1/(slamb *np.sqrt(2*np.pi))
        *np.exp(-0.5*(lamb-lamb0)**2/slamb**2)
        )

    # Error check
    if False:
        print((lamb[1]-lamb[0])/lamb0*100)
        print((1-np.trapz(fE, lamb))*100)
        
        fig, ax = plt.subplots()
        ax.plot(lamb, fE, '*')

    # Output, [AA], [1/AA], dim(nlamb,)
    return lamb, fE

# Interpolating the emissivity data
def _interp_emis(
    spectro=None,
    emis=None,
    eta=None,
    nlamb=None,
    fact=None,
    RR_knots=None,
    ZZ_knots=None,
    RR_asym=None,
    ZZ_asym=None,
    ):

    # Initializes emissivity matrix for on this mesh
    emis2d_init = np.zeros(np.append(RR_knots.shape, nlamb)) # dim(mesh_R, mesh_Z, mesh_lamb)

    # Loop over wavelength
    for ii in np.arange(nlamb):
        emis2d_tmp = emis[:,:,ii*fact] # dim(fm_rhop, fm_theta)

        # Interpolate onto global R,Z mesh
        tmp = griddata((RR_asym.flatten(), ZZ_asym.flatten()),
            emis2d_tmp.flatten(),
            (RR_knots.flatten(), ZZ_knots.flatten()),
            method='linear').reshape(RR_knots.shape) # dim(mesh_R, mesh_Z)

        # Prepares optical efficiency
        if eta.size > 1:
            eff = eta[ii*fact]
        else:
            eff = eta

        # Convert from ph/s/cm3/AA -> ph/s/m3/sr/m
        emis2d_init[:,:,ii] = tmp* 1/(4*np.pi) * 1e6 * 1e10 * eff

    return emis2d_init

