'''

Script to manage organizing certain data

cjperks
Sep 10, 2024

'''

# Module
import numpy as np
import os
from scipy.interpolate import interp1d

import cmp_tofu_xicsrt.utils as utils
import cmp_tofu_xicsrt.setup._def_plasma as dp
import cmp_tofu_xicsrt.setup._def_diag as dd

__all__ = [
    '_add_det_data',
    '_get_dispersion_xicsrt',
    '_build_RZ_mesh_tofu',
    '_build_lamb_mesh_tofu',
    '_build_emis_mesh_tofu',
    ]

############################################
#
#           Detector data
#
############################################

# Adds useful data about the detector
def _add_det_data(
    coll = None,
    key_diag = None,
    key_cam = None,
    dout = None,
    split = True,
    subcam = None,
    ):

    if split:
        kk = key_cam.split('XRS')[0][:-1]
    else:
        kk = key_cam

    # Extent of detector
    dgeom_cam = coll.dobj['camera'][kk]['dgeom']
    out0, out1 = coll.get_optics_outline(kk, total=True)
    dout['extent'] = (
        out0.min(), out0.max(), 
        out1.min(), out1.max()
        )

    # Aspect ratio of detector
    dout['aspect'] = (out0.max()-out0.min())/(out1.max()-out1.min())

    # Pixel centers
    dout['cents_cm'] = [
        (
            dout['extent'][0]
            + (dout['extent'][1]-dout['extent'][0])
            /dout['signal'].shape[0]/2
            + np.arange(dout['signal'].shape[0])
            *(dout['extent'][1]-dout['extent'][0])
            /dout['signal'].shape[0]
            )*100,
        (
            dout['extent'][2]
            + (dout['extent'][3]-dout['extent'][2])
            /dout['signal'].shape[1]/2
            + np.arange(dout['signal'].shape[1])
            *(dout['extent'][3]-dout['extent'][2])
            /dout['signal'].shape[1]
            )*100,
        ]

    if subcam is not None:
        dsub = dd.get_dsubcam(option=subcam)
        if dsub['dx'] is not None:
            dout['cents_cm'] += dsub['dx']*1e2
        if dsub['dy'] is not None:
            dout['cents_cm'] += dsub['dy']*1e2

    # Number of pixels
    dout['npix'] = [
        dout['signal'].shape[0], 
        dout['signal'].shape[1]
        ]

    # Output
    return dout

# Get dispersion data v. pixel from XICSRT
def _get_dispersion_xicsrt(
    dxi = None,         # Direction of XICSRT data from multi_energy run
    lamb0 = None,
    ):

    # Init
    dxi['lambda_pix'] = {}

    # Calculates centroid wavelength
    dxi['lambda_pix']['centroid'] = (
        np.sum(
            dxi['dispersion']*dxi['lambda_AA'][None,None,:],
            axis = -1
            )
        /np.sum(dxi['dispersion'], axis = -1)
        )

    # Calculates resultant Gaussian
    lamb, fE = utils._build_gaussian(lamb0=lamb0)

    dxi['lambda_pix']['gaussian'] = np.zeros_like(dxi['lambda_pix']['centroid'])

    # Loop over vertical pixels
    for yy in np.arange(dxi['lambda_pix']['gaussian'].shape[1]):
        dxi['lambda_pix']['gaussian'][:,yy] = interp1d(
            lamb, fE,
            bounds_error=False,
            fill_value = 0.0
            )(dxi['lambda_pix']['centroid'][:,yy])

    # Output
    return dxi

############################################
#
#           TOFU-specific Mesh data
#
############################################

# Adds emissivity mesh data to Collection object
def _build_emis_mesh_tofu(
    coll = None,
    key_mesh = None,
    key_lamb = None,
    key_emis = None,
    # Data
    emis_RZ = None, # dim(R,Z,lambda); [ph/s/m3/sr/m]
    emis_1d = None, # dim(lambda,); [ph/s/m3/sr/m]
    ):

    # Adds emissivity data
    eunit = 'ph/s/m3/sr/m'
    
    if emis_1d is not None:
        coll.add_data(
            key='emis_1d',
            data=emis_1d,
            ref='%s_bs1'%(key_lamb),
            units=eunit,
            )
    
    coll.add_data(
        key=key_emis,
        data=emis_RZ,
        ref=('%s_bs1'%(key_mesh), '%s_bs1'%(key_lamb)),
        units=eunit,
        )

    # Output
    return coll

# Adds (R,Z) mesh data to Collection object
def _build_RZ_mesh_tofu(
    coll = None,
    key_mesh = None,
    conf = None,
    case = None,
    dplasma = None,
    # Data for flux-function map
    R_knots = None,
    Z_knots = None,
    rhop_RZ = None,
    ):

    # If just want to consider a "square O-ring" shaped plasma
    if case in ['mv', 'me']:

        # Gets default plasma geometry
        if dplasma is None:
            dplasma = dp.get_dplasma(option='default')
            
        # Defines 2D R,Z rectangular mesh
        coll.add_mesh_2d_rect(
            key=key_mesh,
            res=0.01,
            crop_poly=dplasma['crop_poly'],
            deg = 1,
            )

    # If adding flux-function mapping
    elif case in ['rad_emis']:

        # Defines 2D R,Z rectangular mesh
        coll.add_mesh_2d_rect(
            # naming
            key=key_mesh,
            # parameters
            domain=None,
            res=None,            # if res is provided, assumes a uniform mesh, for rectangular use res = [resR, resZ]
            knots0=R_knots,         # if res not provided, can provide the R vector of knots directly
            knots1=Z_knots,         # if res not provided, can provide the Z vector of knots directly
            # optional cropping
            crop_poly=conf,      # cropping of the rectangular mesh with the vessel in the Config
            units  = 'm',
            )

        # Add a 2d bsplines basis on that mesh
        coll.add_bsplines(
            key=key_mesh,
            deg=1
            )

        ########### ----- Add rhop mesh ------ ############

        # Defines flux surfaces
        coll.add_data(
            key='rhop2d',
            data= rhop_RZ,
            ref=('%s_bs1'%(key_mesh))
            )

        '''
        # Add polar mesh based on rhops2d, and associated radial bsplines of degree 2
        nvert  = coll.dobj['camera'][key_cam]['dgeom']['shape'][-1] # Number of pixels in imaging (i.e. vertical) direction
        degbs_1d = 2
        ncam = 1

        nbs_1d = nvert*ncam - degbs_1d

        coll.add_mesh_1d(
            key='m1',
            #knots=np.linspace(0,0.9,20),
            knots=np.linspace(0, 0.90, nbs_1d),
            subkey='rhop2d',
            deg=degbs_1d,
            )
        '''

    # Output
    return coll

# Adds wavelength mesh to Collection object
def _build_lamb_mesh_tofu(
    coll = None,
    key_lamb = None,
    lamb_vec = None,    # [AA]
    ):

    # Adds data to collection object
    coll.add_mesh_1d(
        key=key_lamb,
        knots=lamb_vec*1e-10,
        deg=1,
        units='m',
        )

    # Output
    return coll