'''

Script to scope local emissivity data within the VOS 
of a diagnostic projected in the poloidal cross-section

cjperks
Nov 7th, 2024

'''

# Modules
import numpy as np
import matplotlib.pyplot as plt
import sys, os

import tofu as tf
sys.path.insert(0,'/home/cjperks/usr/python3modules/eqtools3')
import eqtools
sys.path.pop(0)

import cmp_tofu_xicsrt.setup as setup
import cmp_tofu_xicsrt.utils as utils
import cmp_tofu_xicsrt.setup._def_point as dp

__all__ = [
    'scope_emis'
    ]

###################################################
#
#           Main
#
###################################################

# Plot emissivity
def scope_emis(
    coll_tf = None,
    key_diag = None,
    key_cam = None,
    emis_file = None,
    gfile = None,
    lamb0 = None,
    ):

    ###### ---- Emissivity data ---- ######
    # Init
    if emis_file is None:
        emis_file = os.path.join(
            '/home/cjperks/work',
            'conferences/HTPD24/data',
            '170424_XRSHRKr_FAC_test_SpecEmis.npz'
            ) # He-like Kr, PRD profiles

    # Loads data
    emis = np.load(
        emis_file,
        allow_pickle=True
        )['arr_0'][()]

    # Finds index of wavelength of interest
    yind = np.argmin(abs(
        emis[key_diag]['lambda']['data'] - lamb0
        ))

    ###### ---- Diagnostic data ---- ######
    # Builds ToFu diagnostic
    coll = setup._init_diag(
        coll = coll_tf,
        lamb0 = lamb0,
        )

    # extract keys to R, Z coordinates of polygon definign vos in poloidal cross-section
    pcross0, pcross1 = tf.data._class8_vos_utilities._get_overall_polygons(
        coll, 
        doptics=coll.dobj['diagnostic'][key_diag]['doptics'], 
        key_cam=key_cam, 
        poly='pcross', 
        convexHull=False
        )

    ind = np.r_[np.arange(0, pcross0.size), 0]
    pcross = [pcross0[ind], pcross1[ind]]

    ###### ---- Point source data ---- ######
    dres = dp.get_dres(option=key_diag)

    # If user speficially wants pt to be on the magnetic axis
    # NOTE: per the midplane LOS
    dlos = utils._get_tofu_los(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        lamb0 = lamb0
        )
    print('Closest to Magnetic Axis')
    print(dlos['mag_axis'])

    # Mesh to loop over
    dz = np.linspace(
        dlos['mag_axis'][-1] - dres['dz'], 
        dlos['mag_axis'][-1] + dres['dz'], 
        dres['nz']
        ) # dim(nz,)
    dres['pts'] = np.array([dlos['mag_axis'].copy() for _ in dz]) # dim(nz,3)
    dres['pts'][:,-1] = dz # dim(nz,3)

    # Get LOS in (R,Z)
    tt = np.linspace(0,2,501)
    ps = dlos['los_p0'][None,:] - tt[:,None]*dlos['los_vect'][None,:] # dim(npt, 3)
    rs = np.sqrt(ps[:,0]**2 + ps[:,1]**2) # dim(npt,)

    ###### ---- Equilibrium data ---- ######
    if gfile is None:
        gfile = os.path.join(
            '/home/cjperks',
            'tofu_sparc/background_plasma',
            'PRD_plasma/run1',
            'input.geq'
            ) # PRD plasma

    # Reads gfile
    edr = eqtools.EqdskReader(
        gfile=gfile,
        afile=None
        )

    # Gathers data
    dedr = {}

    dedr['rGrid'] = edr.getRGrid()
    dedr['zGrid'] = edr.getZGrid()
    dedr['wall_R'] = edr.getMachineCrossSection()[0]
    dedr['wall_Z'] = edr.getMachineCrossSection()[1]

    dedr['RLCFS'] = edr.getRLCFS()[0]
    dedr['ZLCFS'] = edr.getZLCFS()[0]
    dedr['Rmag'] = edr.getMagR()[0]
    dedr['Zmag'] = edr.getMagZ()[0]

    edr_psiRZ = edr.getFluxGrid()[0]
    edr_psiLCFS = edr.getFluxLCFS()[0]
    edr_psi0 = edr.getFluxAxis()[0]

    #print(min(edr_psiRZ.flatten()))
    #print(edr_psi0)
    #print(edr_psiLCFS)
    #dedr['rhop2D'] = np.sqrt((edr_psiRZ-edr_psi0)/(edr_psiLCFS-edr_psi0))
    dedr['rhop2D'] = np.sqrt(edr_psiRZ/(0-edr_psiLCFS))

    ###### ---- Plotting ---- ######

    fig, ax = plt.subplots(figsize=(10,6))

    # Emissivity
    con = ax.contourf(
        emis['plasma']['rhop_contours']['R']['data'],
        emis['plasma']['rhop_contours']['Z']['data'],
        emis[key_diag]['emis']['data'][:,:,yind],
        levels = 20
        )

    cbar = fig.colorbar(con, ax=ax)
    cbar.set_label(r'$\varepsilon$ [$%s$]'%(emis[key_diag]['emis']['units']))

    # Machine
    ax.plot(
        dedr['wall_R'], 
        dedr['wall_Z'],
        'k',
        linewidth=3,
        alpha = 0.3
        )

    ax.plot(
        dedr['Rmag'], 
        dedr['Zmag'],
        'k+',
        markersize=15,
        markeredgewidth = 3
        )

    rhos = ax.contour(
        dedr['rGrid'],dedr['zGrid'],
        dedr['rhop2D'],
        [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0],
        linestyles='solid',
        #linewidth=2,
        zorder=2,
        colors= 'w'
        #cmap = 'hsv'
        )
    ax.clabel(rhos, inline=True, fontsize = 12)

    ax.plot(
        dedr['RLCFS'], 
        dedr['ZLCFS'],
        'r',
        linewidth=3
        )

    # Diagnostic VOS
    ax.plot(
        pcross[0],pcross[1],
        'k',
        linewidth = 3
        )

    # Spatial-resolution point mesh
    ax.plot(
        np.sqrt(dlos['mag_axis'][0]**2 + dlos['mag_axis'][1]**2),
        dlos['mag_axis'][-1],
        'b*',
        markersize = 18
        )
    ax.plot(
        rs,
        ps[:,-1],
        'b-',
        linewidth = 2
        )

    ax.plot(
        np.sqrt(dres['pts'][:,0]**2 + dres['pts'][:,1]**2),
        dres['pts'][:,-1],
        'r*',
        markersize = 10
        )

    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')

    ax.set_ylim(
        1.2*np.min(pcross[1]),
        1.2*np.max(pcross[1])
        )
    ax.set_xlim(1.2,2.5)

    fig.suptitle(key_cam)
