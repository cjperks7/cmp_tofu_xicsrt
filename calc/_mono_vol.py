'''

Script to handle simulating a monochromatic, volume source

'''

# Modules
import xicsrt
import tofu as tf

import numpy as np
import matplotlib.pyplot as plt
import sys, os

import cmp_tofu_xicsrt.utils as utils
import cmp_tofu_xicsrt.setup as setup

__all__ = [
    'run_mono_vol',
    '_run_mono_vol_xicsrt',
    '_run_mono_vol_tofu',
    ]


###################################################
#
#           Main
#
###################################################

# Simulates a monochromatic, volume source ...
def run_mono_vol(
    coll = None,
    key_diag = None,
    key_cam = None,
    config = None,
    dvol = None,
    subcam = None,
    #vol_plt = vol_plt,
    lamb0 = None,
    # HPC controls
    run_xicsrt = False,
    run_tofu = True,
    dHPC = None,
    dsave = None,
    ):

    # Init
    dout = {}

    # Adds mesh data to compute VOS on
    coll = utils._add_mesh_data(
        coll = coll,
        case = 'simple',
        lamb = lamb0
        )

    # Runs XICSRT
    if run_xicsrt:
        dout['XICSRT'] = _run_mono_vol_xicsrt(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            config = config,
            dvol = dvol,
            dHPC = dHPC,
            lamb0 = lamb0
            )

    # Runs ToFu
    if run_tofu:
        dout['ToFu'], _ = _run_mono_vol_tofu(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            lamb0 = lamb0,
            subcam = subcam,
            )

    # Saves XICSRT data
    if dsave is not None:
        utils._save(
            dout = dout,
            case = 'mv',
            lamb0 = lamb0,
            dsave = dsave,
            dHPC = dHPC
            )
    
    # Output
    return dout


###################################################
#
#           Simulations
#
###################################################

# ... in XICSRT
def _run_mono_vol_xicsrt(
    coll = None,
    key_diag = None,
    key_cam = None,
    config = None,
    dvol = None,
    dHPC = None,
    calc_signal = True,
    #vol_plt = vol_plt,
    lamb0 = None,
    demis = None,
    case = 'mv',
    ):

    # Builds box spatial distribution
    box_cent, box_dl, box_vect = setup._build_boxes(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        lamb0 = lamb0,
        dvol = dvol
        )

    # Builds box angular distribution
    omega_norm, omega_vert, omega_binorm, omega_dl = setup._build_omegas(
        config = config,
        box_cent = box_cent,
        box_vect = box_vect,
        )

    # Partition simulation volume over job arrays
    dout = _loop_volumes_HPC(
        config = config,
        lamb0 = lamb0,
        # Box geometry
        box_cent = box_cent,
        box_vect = box_vect,
        box_dl = box_dl,
        # Emission cone geometry
        omega_norm = omega_norm,
        omega_vert = omega_vert,
        omega_binorm = omega_binorm,
        omega_dl = omega_dl,
        # HPC controls
        dHPC = dHPC,
        calc_signal = calc_signal,
        demis = demis,
        case = case,
        )

    if calc_signal:
        # Stores detector configuration
        dout = utils._add_det_data(
            coll = coll,
            key_diag = key_diag,
            key_cam = key_cam,
            dout = dout
            )

    # Output
    return dout

# ... in ToFu
def _run_mono_vol_tofu(
    coll = None,
    key_diag = None,
    key_cam = None,
    key_mesh = 'mRZ',
    subcam = None,
    # Wavelength controls
    lamb0 = None, # [AA]
    lamb_vec = None,
    # Volume discretization controls
    n0 = 301,
    n1 = 201,
    ):

    # Init
    dout = {}
    print(n0)
    print(n1)

    # Gets wavelength vector to compute VOS on
    if lamb_vec is None:
        lamb_vec = coll.ddata['mlamb_bs1_ap']['data']

    # compute vos
    dvos, dref = coll.compute_diagnostic_vos(
        key_diag=key_diag,
        key_mesh=key_mesh,
        res_RZ=[0.01, 0.01],         # 0.005 would be better
        res_phi=0.0005,        # 0.0002 would be better
        lamb=lamb_vec,
        n0=n0,
        n1=n1,
        visibility=False,
        config=tf.load_config('SPARC-V0'),
        return_vector=False,
        keep3d=False,
        store=True,
        )

    if lamb0 is not None:
        # Extracts VOS of wavelength of interest
        indlamb0 = np.argmin(np.abs(dvos[key_cam]['lamb']['data'] - lamb0*1e-10))
        sig = np.sum(dvos[key_cam]['ph']['data'][:, :, :, indlamb0], axis=2)

        # Saves data
        dout['signal'] = sig/(4*np.pi)*1e6 # dim(nx,ny), [cm^3]
        #dout['dvos'] = dvos[key_cam]

    # extract keys to R, Z coordinates of polygon definign vos in poloidal cross-section
    pcross0, pcross1 = tf.data._class8_vos_utilities._get_overall_polygons(
        coll, 
        doptics=coll.dobj['diagnostic'][key_diag]['doptics'], 
        key_cam=key_cam, 
        poly='pcross', 
        convexHull=False
        )
    phor0, phor1 = tf.data._class8_vos_utilities._get_overall_polygons(
        coll, 
        doptics=coll.dobj['diagnostic'][key_diag]['doptics'], 
        key_cam=key_cam, 
        poly='phor', 
        convexHull=False
        )

    ind = np.r_[np.arange(0, pcross0.size), 0]
    dout['pcross'] = [pcross0[ind], pcross1[ind]]
    ind = np.r_[np.arange(0, phor0.size), 0]
    dout['phor'] = [phor0[ind], phor1[ind]]

    # Stores detector configuration
    dout = utils._add_det_data(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        dout = dout,
        split = False,
        subcam = subcam,
        )
    
    # Output
    return dout, coll

###################################################
#
#           Utilities
#
###################################################

# Loop over volume-of-sight divided for job arrays
def _loop_volumes_HPC(
    config = None, 
    lamb0 = None,   # [AA]
    # Box geometry
    box_cent = None,        # [m], dim(3,norm,vert,binorm), ToFu basis
    box_vect = None,        # [vector], list[norm,vert,binorm], ToFu basis
    box_dl = None,          # [m], list[norm, vert, binorm]
    # Emission cone geometry
    omega_norm = None,      # [vector], dim(3,nnorm,nvert,nbinorm), ToFu basis
    omega_vert = None,      # [vector], dim(3,nnorm,nvert,nbinorm), ToFu basis  
    omega_binorm = None,    # [vector], dim(3,nnorm,nvert,nbinorm), ToFu basis
    omega_dl = None,        # [rad], list[vert,binorm]
    # HPC controls
    dHPC = None,
    calc_signal = None,
    # Velocity controls
    add_velocity = False,
    dvel = None,
    # Local emissivity controls
    demis = None,
    case = 'mv',
    ):

    # Determines how to index the VOS
    axes = dHPC['job_axis']

    j_ax = []   # job axis(es)
    l_ax = []   # looped axis(es)
    jm = 1      # max # of jobs
    lm = 1      # max # of elements in loop
    js = []     # job axis(es) size
    ls = []     # looped axis(es) size
    for ii,ax in enumerate(['norm', 'vert', 'binorm']):
        if ax in axes:
            j_ax.append(ii+1)
            jm *= box_cent.shape[j_ax[-1]]
            js.append(box_cent.shape[j_ax[-1]])

        else:
            l_ax.append(ii+1)
            lm *= box_cent.shape[l_ax[-1]]
            ls.append(np.arange(box_cent.shape[l_ax[-1]]))

    # Error check
    if jm != dHPC['job_max']+1:
        print('Axis mismatch with job array')
        sys.exit(1)

    if len(js) == 2:
        if _have_common_divisor(js[0], js[1]):
            print('Two job axes have GCD. Breaks looping algorithm')
            sys.exit(1)
    elif len(js) >2:
        print('Define a volume axis to loop over')
        sys.exit(1)

    if len(ls) == 2:
        if _have_common_divisor(len(ls[0]), len(ls[1])):
            print('Two looped axes have GCD. Breaks looping algorithm')
            sys.exit(1)
    

    # Creates array of indices to simulate
    inds = np.zeros((3, lm), dtype='int')
    cnt = 0
    for ii in np.arange(inds.shape[-1]):
        for ll, ee in enumerate(l_ax):
            inds[ee-1,ii] = ls[ll][ii%len(ls[ll])]

    for ii,jj in enumerate(j_ax):
        inds[jj-1,:] = dHPC['job_num']%js[ii]

    # Init
    dout = {}
    dout['source'] = {}
    dout['crystal'] = {}
    dout['detector'] = {}

    dout['source']['origin'] = np.empty((0,3)) # dim(nfound, 3)
    dout['crystal']['origin'] = np.empty((0,3)) # dim(nfound, 3)
    dout['detector']['origin'] = np.empty((0,3)) # dim(nfound, 3)

    dout['source']['direction'] = np.empty((0,3)) # dim(nfound, 3)
    dout['crystal']['direction'] = np.empty((0,3)) # dim(nfound, 3)

    # Loops over voxels
    for ii in np.arange(inds.shape[-1]):
        if np.isnan(np.mean(box_cent[:,inds[0,ii], inds[1,ii], inds[2,ii]])):
            continue
        else:
            print(ii)

        # Builds source at this voxel
        config_tmp = setup._build_source(
            config =config,
            lamb0 = lamb0,
            box_cent = box_cent[:,inds[0,ii], inds[1,ii], inds[2,ii]],
            box_vect = box_vect,
            box_dl = box_dl,
            omega_norm = omega_norm[:,inds[0,ii], inds[1,ii], inds[2,ii]],
            omega_vert = omega_vert[:,inds[0,ii], inds[1,ii], inds[2,ii]],
            omega_binorm = omega_binorm[:,inds[0,ii], inds[1,ii], inds[2,ii]],
            omega_dl = omega_dl,
            dHPC = dHPC,
            add_velocity = add_velocity,
            dvel = dvel,
            )

        # Runs ray-tracing
        results = xicsrt.raytrace(config_tmp)

        if False:
            import xicsrt.visual.xicsrt_2d__matplotlib as xicsrt_2d
            xicsrt_2d.plot_intersect(results, 'port3')
            xicsrt_2d.plot_intersect(results, 'ap03')
            xicsrt_2d.plot_intersect(results, 'ap13')
            xicsrt_2d.plot_intersect(results, 'crystal')
            xicsrt_2d.plot_intersect(results, 'detector')

            import pdb
            pdb.set_trace()

        # Stores results
        dout['source']['origin'] = np.vstack((
            results['found']['history']['source']['origin'],
            dout['source']['origin']
            ))
        dout['crystal']['origin'] = np.vstack((
            results['found']['history']['crystal']['origin'],
            dout['crystal']['origin']
            ))
        dout['detector']['origin'] = np.vstack((
            results['found']['history']['detector']['origin'],
            dout['detector']['origin']
            ))

        dout['source']['direction'] = np.vstack((
            results['found']['history']['source']['direction'],
            dout['source']['direction']
            ))
        dout['crystal']['direction'] = np.vstack((
            results['found']['history']['crystal']['direction'],
            dout['crystal']['direction']
            ))

        # Stores voxel data
        dout['voxels'] = {}
        dout['voxels']['dOmega_ster'] = (
            2 *config_tmp['sources']['source']['spread'][0]
            *2 *config_tmp['sources']['source']['spread'][1]
            ) # [ster]
        dout['voxels']['dvol_cm3'] = (
            config_tmp['sources']['source']['zsize']
            *config_tmp['sources']['source']['ysize']
            *config_tmp['sources']['source']['xsize']
            )*1e6 # [cm3]
        dout['voxels']['num_rays'] = (
            config_tmp['general']['number_of_iter']
            *config_tmp['sources']['source']['intensity']
            )

        # Calculates detector signal
        if calc_signal:
            dout = utils._calc_signal(
                dout = dout,
                config = config,
                demis = demis,
                box_cent = box_cent[:,inds[0,ii], inds[1,ii], inds[2,ii]],
                det_origin = results['found']['history']['detector']['origin'],
                case = case,
                )

    '''
    if calc_signal:
        # Calculates histogram
        dhist = utils._calc_det_hist(
            rays = dout['detector']['origin'],
            config = config,
            ) # dim(hor_pix, vert_pix)
        # wavelength-integration (via left-hand Riemann sum)
        dout['signal'] = (
            dhist['counts'] 
            / utils._conv_2normEmis(voxels=dout['voxels'], case='mv')
            ) # dim(nx,ny), [photons/bin^2]
    '''

    # Output
    return dout

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _have_common_divisor(num1, num2):
    if _gcd(num1, num2) > 1:
        return True
    else:
        return False

