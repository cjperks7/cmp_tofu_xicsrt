'''

Script to handle simulating a monochromatic, point source


'''

# Modules
import xicsrt

import numpy as np
import matplotlib.pyplot as plt

import cmp_tofu_xicsrt.utils as utils
import cmp_tofu_xicsrt.setup._def_point as dp
import cmp_tofu_xicsrt.setup as setup

__all__ = [
    'run_mono_pt',
    '_run_mono_pt_xicsrt',
    '_run_mono_pt_tofu',
    ]

###################################################
#
#           Main
#
###################################################

# Simulates a point source ...
def run_mono_pt(
    config = None,
    coll = None,
    key_diag = None,
    key_cam = None,
    cry_shape = None,
    # Controls
    dpt = None,
    lamb0 = None, # [AA]
    run_xicsrt = True,
    run_tofu = True,
    dsave = None,
    # Velocity controls
    add_velocity = False,
    dvel = None,
    ):
    '''
    dpt is a dictionary with keys
        1) 'ToFu':
            i) 'point' -> ToFu x,y,z coordinates of point source
            ii) 'n0' -> horizontal sampling resolution
            iii) 'n1' -> vertical sampling resolution
            iv) 'plt' -> ToFu plot
        2) 'XICSRT'
            i) 'intensty' -> Number of rays to launch
            ii) 'dOmega' -> [horiz., vert.] half-angle angular spread

    '''

    # Init
    dout = {}

    # Gets default values
    if dpt is None:
        if key_diag == 'valid':
            dpt = dp.get_dpt(option=cry_shape)
        else:
            dpt = dp.get_dpt(option=key_diag)

    # Makes the point source
    if dpt['ToFu']['point'] is None:
        # Gets LOS from coll
        vx, vy, vz = coll.get_rays_vect(key_diag)
        ptsx, ptsy, ptsz = coll.get_rays_pts(key_diag)

        xind = dpt['Delta']['xind']
        yind = dpt['Delta']['yind']

        # Defines axis
        norm = np.r_[
            vx[-1, xind, yind], 
            vy[-1, xind, yind], 
            vz[-1, xind, yind]
            ]
        vert = np.r_[0,0,1]
        binorm = np.cross(norm, vert)

        # Point on the wall
        wall = np.r_[
            ptsx[-1,xind,yind], 
            ptsy[-1,xind,yind], 
            ptsz[-1,xind,yind]
            ]

        # Point on Magnetic axis, Hardcoded...................
        origin = wall -norm*0.58

        # Defines point in ToFu terminology
        dpt['ToFu']['point'] = origin + (
            dpt['Delta']['norm']*norm
            + dpt['Delta']['vert']*vert
            + dpt['Delta']['binorm']*binorm
            )

    # Runs XICSRT
    if run_xicsrt:
        dout['XICSRT'] = _run_mono_pt_xicsrt(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            config = config,
            dpt = dpt,
            lamb0 = lamb0,
            add_velocity = add_velocity,
            dvel = dvel,
            )

    # Runs ToFu
    if run_tofu:
        dout['ToFu'] = _run_mono_pt_tofu(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            dpt = dpt,
            lamb0 = lamb0
            )

    # Saves XICSRT data
    if dsave is not None:
        utils._save(
            dout = dout,
            case = 'pt',
            lamb0 = lamb0,
            dsave = dsave,
            )

    # Output
    return dout

###################################################
#
#           Utilities
#
###################################################

# ... in XICSRT
def _run_mono_pt_xicsrt(
    coll = None,
    key_diag = None,
    key_cam = None,
    config = None,
    dpt = None,
    lamb0 = None, # [AA]
    pt_plt = False,
    # Detector mesh
    #nx = 256,
    #ny = 64,
    nx = 1028,
    ny = 1062,
    # Velocity control
    add_velocity = False,
    dvel = None,
    ):

    # Init
    dout = {}

    # Defines point source origin
    pt = utils._tofu2xicsrt(
        data = dpt['ToFu']['point']
        )
    #pt = np.r_[1.8, 0.18, 0]
    #pt = np.r_[0.18, 0, 1.8]
    #pt = np.r_[0.18,0,1.8]
    
    # Defines normal axis of point source toward crystal center
    # extract dict of optics
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    kap = doptics[key_cam]['optics'][1:][0]
    #vpt = config['optics']['crystal']['origin'] - pt
    vpt = config['optics'][kap]['origin'] - pt
    vpt /= np.linalg.norm(vpt)
    #print(vpt)

    # Defines the vertical and binormal directions
    vert = [
        -1,
        vpt[0]/vpt[1],
        0
        ]
    vert /= np.linalg.norm(vert)
    binorm = np.cross(vert, vpt)

    # Init
    config['sources'] = {}
    config['sources']['source'] = {} # Only one source is supported at this time

    # Source origin
    config['sources']['source']['origin'] = pt

    # Source orientation
    config['sources']['source']['zaxis'] = vpt
    config['sources']['source']['xaxis'] = vert
    #config['sources']['source']['xaxis'] = binorm

    _, _, _, omega_dl = setup._build_omegas(
        config = config,
        box_cent = dpt['ToFu']['point'][:,None,None,None],
        box_vect = [
            utils._xicsrt2tofu(vpt),
            utils._xicsrt2tofu(vert),
            utils._xicsrt2tofu(binorm)
            ],
        )
    dpt['XICSRT']['dOmega'] = [
        1.1*np.max(abs(omega_dl[0,:])),
        1.1*np.max(abs(omega_dl[1,:]))
        ]

    # Source type
    config['sources']['source']['class_name'] = 'XicsrtSourceDirected'

    # Number of rays to emit
    config['sources']['source']['intensity'] = dpt['XICSRT']['intensity']

    # Wavelength distirbution
    config['sources']['source']['wavelength_dist'] = 'monochrome'
    config['sources']['source']['wavelength'] = lamb0 # [AA]

    # Angular distribution
    config['sources']['source']['angular_dist'] = 'isotropic_xy' # Horizontal & Vertical extent
    config['sources']['source']['spread'] = dpt['XICSRT']['dOmega'] # [rad], (x,y), half-angles

    # If adding Doppler shift
    if add_velocity:
        config['sources']['source']['velocity'] = utils._tofu2xicsrt(
            data = setup._add_velocity(
                dvel = dvel,
                #box_cent = dpt['ToFu']['point'][:,None,None,None]
                box_cent = dpt['ToFu']['point'],
                )
            ) # [m/s], clockwise toroidal flow

    # Runs ray-tracing
    results = xicsrt.raytrace(config)
    #dxicsrt['pt']['XICSRT'] = results['found']['history']['detector']
    dout['results'] = results

    # Plots intersect of point source rays with Port
    if pt_plt:
        import xicsrt.visual.xicsrt_2d__matplotlib as xicsrt_2d
        xicsrt_2d.plot_intersect(results, kap)
        xicsrt_2d.plot_intersect(results, 'crystal')
        xicsrt_2d.plot_intersect(results, 'detector')

    # Calculates histogram
    dhist = utils._calc_det_hist(
        rays = results['found']['history']['detector']['origin'],
        config = config,
        nx = nx,#128,
        ny = ny
        ) # dim(hor_pix, vert_pix)
    # wavelength-integration (via left-hand Riemann sum)
    voxels = {
        'num_rays': dpt['XICSRT']['intensity'] * config['general']['number_of_iter'],
        'dOmega_ster': (
            2*dpt['XICSRT']['dOmega'][0]
            *2*dpt['XICSRT']['dOmega'][1]
            )
        }
    dout['signal'] = (
        dhist['counts'] 
        / utils._conv_2normEmis(voxels=voxels, case='pt')
        ) # dim(nx,ny), [photons/bin^2]
    dout['config'] = config

    # Stores detector configuration
    dout = utils._add_det_data(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        dout = dout,
        split = True,
        )

    # Output
    return dout

# .... in ToFu
def _run_mono_pt_tofu(
    coll = None,
    key_diag = None,
    key_cam = None,
    dpt = None,
    lamb0 = None,
    ):

    # Init
    dout = {}

    # Runs ToFu point-source ray-tracing
    dout['ray-trace'] = coll.get_raytracing_from_pts(
        key=key_diag,
        ptsx=np.r_[dpt['ToFu']['point'][0]],
        ptsy=np.r_[dpt['ToFu']['point'][1]],
        ptsz=np.r_[dpt['ToFu']['point'][2]],
        n0=dpt['ToFu']['n0'],
        n1=dpt['ToFu']['n1'],
        lamb0=lamb0*1e-10, # [m]
        rocking_curve=True,
        #rocking_curve=False,
        # plot
        plot=dpt['ToFu']['plt'],
        )

    # Calculates signal from 1ph/s source
    dout['signal'] = dout['ray-trace']['sang_lamb']['data']/(4*np.pi)

    # Stores detector configuration
    dout = utils._add_det_data(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        dout = dout,
        split = False,
        )

    # Output
    return dout

