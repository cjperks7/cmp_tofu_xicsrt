'''

Script to scope the effects of Doppler shift

cjperks
Dec 10th, 2024

'''

# Modules
import sys, os
import numpy as np

import tofu as tf

from transport_world.run_profiletools import eqtools3 as eq

# Init
key_diag = 'XRSHRKr'
key_cam = 'e1M3_'+key_diag
lamb0 = 0.945 # [AA]

# SPARC Geom
R0 = 1.85 # [m]
aa = 0.57 # [m]

# Collection object
coll = tf.data.load(
    os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt/diags',
        'sparc_htpd24_v3.npz'
        )
    )

# Gets the tangency radius
R_tan = coll.get_rays_quantity(
    key=key_diag+'_'+key_cam+'_los',
    quantity = 'tangency_radius',
    segment=-1,
    lim_to_segments=False,
    )[0] # dim(nx, ny)

# Gets wavelength mesh
lamb, refs = coll.get_diagnostic_lamb(
    key_diag,
    key_cam=key_cam,
    lamb='lamb',
    ) # dim(nx, ny)

# Gets LOS point/vector data
ptsx, ptsy, ptsz = coll.get_rays_pts(key_diag)
vx, vy, vz = coll.get_rays_vect(key_diag)

# Indexing
indy = int(lamb.shape[1]/2-1) # Takes midline
indx = np.nanargmin(abs(lamb0*1e-10 - lamb[:,indy]))

R_tan0 = R_tan[indx,indy]

pts = np.c_[ptsx[:,indx,indy], ptsy[:,indx,indy],  ptsz[:,indx,indy]]
v_LOS = np.r_[vx[-1,indx,indy], vy[-1,indx,indy],vz[-1,indx,indy]] # LOS vector

# LOS trajectory
rr = np.linspace(-2,2,501) # [m]
traj = pts[-1,:] - rr[:,None]*v_LOS[None,:] # dim(npt, 3)

# Finds the height at the tangency radius
indr = np.argmin(abs(
    np.sqrt(traj[:,0]**2+traj[:,1]**2) - R_tan0
    ))

# Tangency height
Z_tan = traj[indr,-1]

# Inclination angle
Psi = np.arctan(
    (Z_tan - traj[0,-1])
    /np.sqrt(
        traj[0,0]**2 + traj[0,1]**2
        - R_tan0**2
        )
    )

# Calculates the toroidal part of the LOS per Matt's THACO equations
lphi_THACO = (
    np.cos(Psi)
    * R_tan0
    /np.sqrt(traj[:,0]**2+traj[:,1]**2)
    )

# Calculates the toroidal part from the trajectory
lphi_traj = np.zeros_like(lphi_THACO)
for ii in np.arange(lphi_traj.shape[0]):
    R_star = np.r_[traj[ii,0], traj[ii,1],0]
    R_star /= np.linalg.norm(R_star)

    Z_star = np.r_[0,0,1]
    phi_star = np.cross(Z_star, R_star)

    lphi_traj[ii] = np.dot(phi_star, v_LOS)

# Gets plasma equilibrium
in_path = os.path.join(
    '/home/cjperks',
    'tofu_sparc/background_plasma',
    'PRD_plasma/run1'
    )
dedr, edr = eq._get_eq(
    gfile = in_path+'/input.geq',
    afile = None,
    #afile = in_path+'/workAround.aeq',
    machine = 'SPARC'
    )





# Plotting
fig, ax = plt.subplots(2,2, figsize=(12,12))
ax[0,0].plot(
    traj[:,0], traj[:,1],
    color = 'blue',
    label = 'LOS'
    )

theta = np.linspace(0, np.pi/2, 501)
ax[0,0].plot(
    R0*np.cos(theta), R0*np.sin(theta),
    '--',
    color = 'm',
    label = 'plasma'
    )
ax[0,0].plot(
    (R0-aa)*np.cos(theta), (R0-aa)*np.sin(theta),
    '-',
    color = 'm',
    label = 'plasma'
    )
ax[0,0].plot(
    (R0+aa)*np.cos(theta), (R0+aa)*np.sin(theta),
    '-',
    color = 'm',
    label = 'plasma'
    )

ax[0,0].plot(
    [0,traj[indr,0]],
    [0, traj[indr,1]],
    '-',
    color = 'r',
    label = 'tangency radius'
    )

ax[0,0].set_aspect('equal')
ax[0,0].grid('on')
ax[0,0].set_xlabel('X [m]')
ax[0,0].set_ylabel('Y [m]')

ax[0,1].plot(
    np.sqrt(traj[:,0]**2 + traj[:,1]**2),
    traj[:,2],
    '-',
    color = 'blue',
    label = 'LOS'
    )

ax[0,1].plot(
    [0,np.sqrt(traj[indr,0]**2+traj[indr,1]**2)],
    [0, traj[indr,2]],
    '-',
    color = 'r',
    label = 'tangency radius'
    )    

ax[0,1].plot(dedr['RLCFS'], dedr['ZLCFS'],'m',linewidth=3)
con = ax[0,1].contour(
    dedr['rGrid_1d'],dedr['zGrid_1d'],
    dedr['rhop2D'],
    [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0],
    linestyles='solid',
    linewidth=2,
    zorder=2,
    #cmap = 'hsv'
    colors = 'm'
    )
ax[0,1].clabel(con, inline=True, fontsize = 8)

ax[0,1].set_aspect('equal')
ax[0,1].grid('on')
ax[0,1].set_xlabel('R [m]')
ax[0,1].set_ylabel('Z [m]')


ax[1,0].plot(
    np.sqrt(traj[:,0]**2+traj[:,1]**2),
    lphi_THACO,
    color = 'r',
    label = 'Matt/THACO'
    )
ax[1,0].plot(
    np.sqrt(traj[:,0]**2+traj[:,1]**2),
    lphi_traj,
    color = 'b',
    label = 'trajectory'
    )

ax[1,0].plot(
    [R0, R0],    
    [0, 1],
    '--',
    color = 'm',
    #label = 'plasma'
    )
ax[1,0].plot(
    [np.min(dedr['RLCFS']), np.min(dedr['RLCFS'])],
    [0, 1],
    '-',
    color = 'm',
    #label = 'plasma'
    )
ax[1,0].plot(
    [np.max(dedr['RLCFS']), np.max(dedr['RLCFS'])],    
    [0, 1],
    '-',
    color = 'm',
    #label = 'plasma'
    )

ax[1,0].set_xlabel('R [m]')
ax[1,0].set_ylabel(r'$\hat{l}_{\phi}$')
ax[1,0].grid('on')

leg = ax[1,0].legend(labelcolor='linecolor', loc = 'upper right')




ax[1,1].plot(
    np.sqrt(traj[:,0]**2+traj[:,1]**2),
    lphi_THACO*np.sqrt(traj[:,0]**2+traj[:,1]**2),
    color = 'r',
    label = 'Matt/THACO'
    )
ax[1,1].plot(
    np.sqrt(traj[:,0]**2+traj[:,1]**2),
    lphi_traj*np.sqrt(traj[:,0]**2+traj[:,1]**2),
    color = 'b',
    label = 'trajectory'
    )

ax[1,1].plot(
    [R0, R0],    
    [0, 1],
    '--',
    color = 'm',
    #label = 'plasma'
    )
ax[1,1].plot(
    [np.min(dedr['RLCFS']), np.min(dedr['RLCFS'])],
    [0, 1],
    '-',
    color = 'm',
    #label = 'plasma'
    )
ax[1,1].plot(
    [np.max(dedr['RLCFS']), np.max(dedr['RLCFS'])],    
    [0, 1],
    '-',
    color = 'm',
    #label = 'plasma'
    )

ax[1,1].set_xlabel('R [m]')
ax[1,1].set_ylabel(r'R$\hat{l}_{\phi}$ [m]')
ax[1,1].grid('on')

