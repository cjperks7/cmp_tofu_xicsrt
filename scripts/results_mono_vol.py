'''

Script to read results of a mono-energetci, volumetric source simulation

cjperks
Aug 26, 2024

'''

# Modules
import numpy as np
import os, sys
import cmp_tofu_xicsrt.plotting as plotting
import cmp_tofu_xicsrt.utils as utils

# Enables automatic reloading of modules
%reload_ext autoreload
%autoreload 2


# Folder to mv results
fol = os.path.join(
    '/home/cjperks/cmp_tofu_xicsrt',
    'HPC/output'
    )
fol_xi = 'cyl_mv_v2'
fol_tf = 'cyl_mv_v1'
name = 'delta_cyl'


# Extracts HPC data
ddata = utils._get_mv_results(
    folder = fol,
    folder_xi = fol_xi,
    folder_tf = fol_tf,
    name = name
    )


# Plots results
plotting.plt_mono_vol(
    ddata = ddata,
    )








'''

sys.path.insert(0,'/home/cjperks/usr/python3modules/eqtools3')
import eqtools
sys.path.pop(0)


# Reads SPARC eqdsk
edr = eqtools.EqdskReader(
    gfile = os.path.join(
        '/home/cjperks/tofu_sparc',
        'background_plasma/PRD_plasma',
        'run1/input.geq',
        ),
    afile = None
    )

edr_RLCFS = edr.getRLCFS()[0]
edr_ZLCFS = edr.getZLCFS()[0]

edr_mach = edr.getMachineCrossSection()


# Generates tokamak as a circle
theta = np.linspace(0, 2*np.pi, 1001)

rad_w0 = np.min(edr_mach[0])
rad_w1 = np.max(edr_mach[0])

X_w0 = rad_w0 *np.cos(theta)
Y_w0 = rad_w0 *np.sin(theta)

X_w1 = rad_w1 *np.cos(theta)
Y_w1 = rad_w1 *np.sin(theta)

Zmax = 9
Zmin = -5
R0 = 1.85*100
aa = 0.57*100

X1 = (R0-aa)*np.cos(theta)
Y1 = (R0-aa)*np.sin(theta)

X2 = (R0+aa)*np.cos(theta)
Y2 = (R0+aa)*np.sin(theta)


box_cent, box_dl, box_vect = setup._build_boxes(
        coll = coll,
        key_diag = 'valid',
        key_cam = 'cam',
        lamb0 = 1.61,
        dvol = None
        )






jjs = ['16']
fig, ax = plt.subplots(2,2, figsize=(10,10))
plt.rcParams.update({'font.size': 14})
plt.subplots_adjust(wspace=0.4,hspace=0.4)
lw = 2

ax[0,0].plot(
    edr_mach[0]*100,
    edr_mach[1]*100,
    'k-',
    linewidth=lw
    )

ax[0,0].plot(
    edr_RLCFS*100,
    edr_ZLCFS*100,
    'm-',
    linewidth=lw
    )


#for jj in range(1):
#for jj in dxi.keys():
for jj in jjs:
    if jj in skip:
        continue
    ax[0,0].plot(
        np.sqrt(
            dxi[jj]['source']['origin'][:,0]**2
            +dxi[jj]['source']['origin'][:,1]**2)*100,
        dxi[jj]['source']['origin'][:,2]*100,
        #'r.'
        '.'
        )

ax[0,0].plot(
    dtf['pcross'][0]*100,
    dtf['pcross'][1]*100,
    'b-'
    )

ax[0,0].plot(
    np.sqrt(box_cent[0,:,:,:].flatten()**2 + box_cent[1,:,:,:].flatten()**2)*100,
    box_cent[2,:,:,:].flatten()*100,
    'k*',
    markersize = 10
    )

ax[0,0].grid('on')
ax[0,0].set_xlabel('R [cm]')
ax[0,0].set_ylabel('Z [cm]')


ax[0,1].plot(X_w0*100, Y_w0*100, 'k-')
ax[0,1].plot(X_w1*100, Y_w1*100, 'k-')

ax[0,1].plot(X1, Y1, 'm-')
ax[0,1].plot(X2, Y2, 'm-')

ax[0,1].set_xlim(125,225)
ax[0,1].set_ylim(5,25)

#for jj in range(1):
#for jj in dxi.keys():
for jj in jjs:
    if jj in skip:
        continue
    ax[0,1].plot(
        dxi[jj]['source']['origin'][:,0]*100,
        dxi[jj]['source']['origin'][:,1]*100,
        #'r.'
        '.'
        )

ax[0,1].plot(
    dtf['phor'][0]*100,
    dtf['phor'][1]*100,
    'b-'
    )

ax[0,1].plot(
    box_cent[0,:,:,:].flatten()*100,
    box_cent[1,:,:,:].flatten()*100,
    'k*',
    markersize = 10
    )

ax[0,1].grid('on')
ax[0,1].set_xlabel('X [cm]')
ax[0,1].set_ylabel('Y [cm]')



#for jj in range(1):
#for jj in dxi.keys():
for jj in jjs:
    if jj in skip:
        continue
    ax[1,0].plot(
        dxi[jj]['source']['origin'][:,0]*100,
        dxi[jj]['source']['origin'][:,2]*100,
        #'r.'
        '.'
        )

ax[1,0].grid('on')
ax[1,0].set_xlabel('X [cm]')
ax[1,0].set_ylabel('Z [cm]')



#for jj in range(1):
#for jj in dxi.keys():
for jj in jjs:
    if jj in skip:
        continue
    ax[1,1].plot(
        dxi[jj]['source']['origin'][:,1]*100,
        dxi[jj]['source']['origin'][:,2]*100,
        #'r.'
        '.'
        )

ax[1,1].grid('on')
ax[1,1].set_xlabel('Y [cm]')
ax[1,1].set_ylabel('Z [cm]')







src_ori = np.zeros(3)
nn = 0
src_los = np.zeros(3)

for jj in dxi.keys():
    if jj in skip:
        continue
    src_ori += np.sum(
        dxi[jj]['source']['origin'],
        axis = 0
        )
    src_los += np.sum(
        dxi[jj]['source']['direction'],
        axis = 0
        )
    nn += dxi[jj]['source']['origin'].shape[0]
    print(jj)

src_ori /= nn
src_los /= nn

tt = np.r_[0,1]
pps = src_ori[None,:] + tt[:,None]*src_los[None,:]

ax[0,1].plot(
    pps[:,0]*100,
    pps[:,1]*100,
    'r-'
    )

key_diag = 'valid'
key_cam = 'cam'
lamb, refs = coll.get_diagnostic_lamb(
    key_diag,
    key_cam=key_cam,
    lamb='lamb',
    )

vx, vy, vz = coll.get_rays_vect(key_diag)
ptsx, ptsy, ptsz = coll.get_rays_pts(key_diag)

ind = np.argmin(
    abs(lamb[:,31]-1.61e-10)
)

v_tf = np.r_[
    vx[-1,ind,31],
    vy[-1,ind,31],
    vz[-1,ind,31]
    ]

p_tf1 = np.r_[
    ptsx[-1,ind,31],
    ptsy[-1,ind,31],
    ptsz[-1,ind,31]
    ]
p_tf2 = np.r_[
    ptsx[-2,ind,31],
    ptsy[-2,ind,31],
    ptsz[-2,ind,31]
    ]

ax[0,1].plot(
    [p_tf1[0]*100, p_tf2[0]*100],
    [p_tf1[1]*100, p_tf2[1]*100],
    'g-'
    )

'''