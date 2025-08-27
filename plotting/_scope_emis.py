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
from scipy.interpolate import griddata, interp1d

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
    print(dlos['los_mag_axis'])

    # Mesh to loop over
    dz = np.linspace(
        dlos['los_mag_axis'][-1] - dres['dz'], 
        dlos['los_mag_axis'][-1] + dres['dz'], 
        dres['nz']
        ) # dim(nz,)
    dres['pts'] = np.array([dlos['los_mag_axis'].copy() for _ in dz]) # dim(nz,3)
    dres['pts'][:,-1] = dz # dim(nz,3)

    # Get LOS in (R,Z)
    tt = np.linspace(0,2,501)
    ps = dlos['los_inboard'][None,:] - tt[:,None]*dlos['los_vect'][None,:] # dim(npt, 3)
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
    dedr['rGrid_2d'], dedr['zGrid_2d'] = np.meshgrid(dedr['rGrid'], dedr['zGrid'])

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
    cbar.set_label(r'$\varepsilon$ [$%s$] @ $\lambda=$%0.3f $\AA$'%(
        emis[key_diag]['emis']['units'], lamb0
        ))

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
        np.sqrt(dlos['los_mag_axis'][0]**2 + dlos['los_mag_axis'][1]**2),
        dlos['los_mag_axis'][-1],
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




    ######### --- Look at LOS v. (rho, Z) ------ ############

    fig, ax = plt.subplots(figsize=(10,6))

    # Gets wavelength mesh
    lamb, refs = coll.get_diagnostic_lamb(
        key_diag,
        key_cam=key_cam,
        lamb='lamb',
        ) # dim(nx, ny)

    # Gets LOS point/vector data
    ptsx, ptsy, ptsz = coll.get_rays_pts(key_diag)
    vx, vy, vz = coll.get_rays_vect(key_diag)

    rho_tan = []
    indr = []
    for ii in np.arange(lamb.shape[-1]):
        try:
            ind = np.nanargmin(abs(lamb[:,ii] - lamb0*1e-10))
            indr.append(ii)
        except:
            continue

        # Get LOS in (R,Z)
        tt = np.linspace(0,2,501)
        ps = (
            np.r_[ptsx[-1,ind,ii], ptsy[-1,ind,ii], ptsz[-1,ind,ii]][None,:] 
            - tt[:,None]*np.r_[vx[-1,ind,ii], vy[-1,ind,ii], vz[-1,ind,ii]][None,:] 
            )# dim(npt, 3)
        rs = np.sqrt(ps[:,0]**2 + ps[:,1]**2) # dim(npt,)

        rhos = griddata(
            (
                dedr['rGrid_2d'].flatten(),dedr['zGrid_2d'].flatten()
                ),
            dedr['rhop2D'].flatten(),
            (rs, ps[:,-1]),
            fill_value = np.nan,
            method='linear') # dim(scalar); [ph/s/cm3/AA]
        rho_tan.append(np.nanmin(rhos))

        ax.plot(
            rs,
            ps[:,-1],
            '-',
            linewidth = 2,
            label = '%0.2f cm'%(coll.ddata['e1M3_'+key_diag+'_c1']['data'][ii]*100)
            )

    leg = ax.legend(labelcolor='linecolor')

    # Emissivity
    con = ax.contourf(
        emis['plasma']['rhop_contours']['R']['data'],
        emis['plasma']['rhop_contours']['Z']['data'],
        emis[key_diag]['emis']['data'][:,:,yind],
        levels = 20
        )

    cbar = fig.colorbar(con, ax=ax)
    cbar.set_label(r'$\varepsilon$ [$%s$] @ $\lambda=$%0.3f $\AA$'%(
        emis[key_diag]['emis']['units'], lamb0
        ))

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

    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')

    ax.set_ylim(
        1.2*np.min(pcross[1]),
        1.2*np.max(pcross[1])
        )
    ax.set_xlim(1.2,2.5)

    fig.suptitle(key_cam)


    fig2, ax2 = plt.subplots(1,2)

    ax2[0].plot(
        emis[key_diag]['grid']['data'][0],
        emis[key_diag]['emis']['data'][:,0,yind],
        'b-',
        )

    for ii in np.arange(len(rho_tan)):
        rr = interp1d(emis[key_diag]['grid']['data'][0],emis[key_diag]['emis']['data'][:,0,yind])(rho_tan[ii])
        ax2[0].plot(
            rho_tan[ii],
            rr,
            '*',
            markersize = 15,
            label = '%0.2f cm'%(coll.ddata['e1M3_'+key_diag+'_c1']['data'][indr[ii]]*100)
            )

    ax2[0].set_xlabel(r'$\sqrt{\Psi_n}$')
    ax2[0].set_ylabel(r'$\varepsilon$ [$%s$] @ $\lambda=$%0.3f $\AA$'%(
        emis[key_diag]['emis']['units'], lamb0
        ))
    ax2[0].grid('on')
    ax2[0].set_xlim(0,0.15)
    ax2[0].set_ylim(2.5e15, 2.7e15)

    ax2[1].plot(
        emis[key_diag]['grid']['data'][0],
        emis['plasma']['Ti']['data']/1e3,
        'b-',
        )

    for ii in np.arange(len(rho_tan)):
        ax2[1].plot(
            rho_tan[ii],
            interp1d(emis[key_diag]['grid']['data'][0],emis['plasma']['Ti']['data']/1e3)(rho_tan[ii]),
            '*',
            markersize = 15,
            label = '%0.2f cm'%(coll.ddata['e1M3_'+key_diag+'_c1']['data'][indr[ii]]*100)
            )

    ax2[1].set_xlabel(r'$\sqrt{\Psi_n}$')
    ax2[1].set_ylabel(r'$T_i$ [keV]')
    ax2[1].grid('on')
    ax2[1].set_xlim(0,0.15)
    ax2[1].set_ylim(17, 21)
    leg = ax2[1].legend(labelcolor='linecolor')
    leg.set_draggable('on')