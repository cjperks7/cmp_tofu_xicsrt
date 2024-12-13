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

        dlamb = demis['dlamb'] # dim(scalar); [AA]
        nlamb = demis['nlamb']
        ilamb = demis['ilamb']

    # Calculates histogram
    dhist = utils._calc_det_hist(
        rays = det_origin,
        config = config,
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

# Formats emissivity data for convenient use in XICSRT
def _prep_emis_xicsrt(
    coll = None,
    key_diag = None,
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
        'mesh_R': coll.ddata['m0_k0']['data'], # dim(mesh_R); [m]
        'mesh_Z': coll.ddata['m0_k1']['data'], # dim(mesh_Z); [m]
        'dlamb': dlamb, # dim(scalar); [AA]
        'nlamb': nlamb,
        'ilamb': ilamb,
        }

# Imoort the emissivity data into Collection object 
def _prep_emis_tofu(
    coll = None,
    key_diag = None,
    key_cam = None,
    emis_file = None,
    conf = None,
    # R,Z discretization
    R_knots = None,
    Z_knots = None,
    # wavelength discretization
    nlamb= 500
    ):

    # Loads data
    emis = np.load(
        emis_file,
        allow_pickle=True
        )['arr_0'][()]

    ########### ----- Add (R,Z) and wavelength data ------ ############

    if conf is None:
        conf = tf.load_config('SPARC-V0')

    # If user wishes to use (R,Z) mesh from geqdsk file
    if R_knots is None:
        R_knots = emis['plasma']['RR']['data'] # [m], dim(R,)
    if Z_knots is None:
        Z_knots = emis['plasma']['ZZ']['data'] # [m], dim(Z,)

    # Number of points to skip on fine wavelength mesh
    fact = int(emis[key_diag]['emis']['data'].shape[-1]/nlamb)

    # Ensures wavelength data is monotonically increasing
    lamb = emis[key_diag]['lambda']['data'][::fact]
    flip = False
    if np.mean(lamb[1:]-lamb[:-1])<0:
        lamb = np.flip(lamb)
        flip = True

    # Adds mesh data
    coll = utils._add_mesh_data(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        conf=conf,
        case='rad_emis',
        R_knots=R_knots,
        Z_knots=Z_knots,
        lamb = lamb, # [AA]
        rhop_RZ=np.sqrt(emis['plasma']['PSIN_RZ']['data'])
        )

    ########### ----- Add emissivity data ------ ############

    # Interpolates from transport grid onto regular grid
    nR, nZ = R_knots.size, Z_knots.size
    RR_knots = np.repeat(R_knots[:, None], nZ, axis=1) # dim(mesh_R, mesh_Z)
    ZZ_knots = np.repeat(Z_knots[None, :], nR, axis=0) # dim(mesh_R, mesh_Z)
    emis2d_tmp_m0 = _interp_emis(
        emis=emis[key_diag]['emis']['data'], # dim(fm_rhop, fm_theta, fm_nlamb)
        eta=emis[key_diag]['eta']['data'], # dim(fm_nlamb)
        nlamb=nlamb,
        fact=fact,
        RR_knots=RR_knots, # dim(mesh_R, mesh_Z)
        ZZ_knots=ZZ_knots, # dim(mesh_R, mesh_Z)
        RR_asym=emis['plasma']['rhop_contours']['R']['data'], # dim(fm_rhop, fm_theta),
        ZZ_asym=emis['plasma']['rhop_contours']['Z']['data'], # dim(fm_rhop, fm_theta)
        ) # dim(mesh_R, mesh_Z, nlamb)

    if flip:
        emis2d_tmp_m0 = np.flip(emis2d_tmp_m0, axis=-1)

    # Add emissivity data
    coll.add_data(
        key='emis_'+key_diag,
        data=emis2d_tmp_m0,
        ref=('m0_bs1','mlamb_'+key_diag+'_bs1'),
        units='ph/s/m3/sr/m',
        )

    # Output
    return coll


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

