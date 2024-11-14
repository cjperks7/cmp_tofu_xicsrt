'''

Function to build voxel discretization of XICSRT volume

cjperks
Aug 5, 2024

'''

# Modules
import numpy as np
import itertools as itt

import cmp_tofu_xicsrt.utils as utils
import cmp_tofu_xicsrt.setup._def_voxels as dv
import cmp_tofu_xicsrt.setup._def_plasma as dp

__all__ = [
    '_build_omegas',
    '_build_boxes'
]


#############################################################
#
#               Main
#
#############################################################


# Defines emission cone solid angle
def _build_omegas(
    config = None,
    box_cent = None, # [m], dim(3, norm, vert, binorm), ToFu basis
    box_vect = None, # [vector], list[norm, vert, binorm]
    # Controls
    method = 'centered', # 'boxes' or 'centered'
    key_ap = None,
    debug = False,
    ):

    # Aperture to focus on
    if key_ap is None:
        optics = list(config['optics'].keys())

        for ii, op in enumerate(optics):
            if op in ['crystal', 'crKr3']:
                ind = ii
        key_ap = optics[int(ind-1)]

    # Aperture geometry
    ap_cent = utils._xicsrt2tofu(data=config['optics'][key_ap]['origin'])
    ap_norm = utils._xicsrt2tofu(data=config['optics'][key_ap]['zaxis'])
    ap_binorm = utils._xicsrt2tofu(data=config['optics'][key_ap]['xaxis'])
    ap_vert = np.cross(ap_norm, ap_binorm)

    # Aperture corners
    ap_tl = (
        ap_cent
        + config['optics'][key_ap]['ysize']/2 * ap_vert
        + config['optics'][key_ap]['xsize']/2 * ap_binorm
        )
    ap_tr = (
        ap_cent
        + config['optics'][key_ap]['ysize']/2 * ap_vert
        - config['optics'][key_ap]['xsize']/2 * ap_binorm
        )
    ap_bl = (
        ap_cent
        - config['optics'][key_ap]['ysize']/2 * ap_vert
        + config['optics'][key_ap]['xsize']/2 * ap_binorm
        )
    ap_br = (
        ap_cent
        - config['optics'][key_ap]['ysize']/2 * ap_vert
        - config['optics'][key_ap]['xsize']/2 * ap_binorm
        )
    
    # Init
    omega_norm = np.nan*np.zeros(box_cent.shape) # dim(3, norm, vert, binorm)
    omega_vert = np.nan*np.zeros(box_cent.shape) # dim(3, norm, vert, binorm)
    omega_binorm = np.nan*np.zeros(box_cent.shape) # dim(3, norm, vert, binorm)
    omega_dl = np.zeros((2,2)) # dim(left/right, vert/binorm)

    # Loop over centers
    for ii in itt.product(
        range(box_cent.shape[1]),range(box_cent.shape[2]), range(box_cent.shape[3])
        ):
        # Error check
        if np.isnan(np.mean(box_cent[:,ii[0], ii[1], ii[2]])):
            continue

        # Defines emission axis
        if method == 'boxes':       # If aligned with boxes
            omega_norm[:,ii[0], ii[1], ii[2]] = -1*box_vect[0]
            omega_vert[:,ii[0], ii[1], ii[2]] = box_vect[1]
            omega_binorm[:,ii[0], ii[1], ii[2]] = -1*box_vect[2]

        elif method == 'centered':  # If aligned with aperture
            norm_new = ap_cent - box_cent[:,ii[0], ii[1], ii[2]]
            norm_new /= np.linalg.norm(norm_new)

            # Compute rotation matrix
            rot_mat = _rotation_matrix_from_vectors(
                -1*box_vect[0], norm_new
                )

            # Stack the basis vectors into a matrix
            basis = np.column_stack((-1*box_vect[0], -1*box_vect[2], box_vect[1]))

            # Rotate the basis vectors
            rotated_basis = np.dot(basis, rot_mat)

            omega_norm[:,ii[0], ii[1], ii[2]] = rotated_basis[:,0]
            omega_vert[:,ii[0], ii[1], ii[2]] = rotated_basis[:,2]
            omega_binorm[:,ii[0], ii[1], ii[2]]  = rotated_basis[:,1]


        # Vector from box centers to aperture corners
        v_tl = ap_tl - box_cent[:,ii[0],ii[1],ii[2]]
        v_tl /= np.linalg.norm(v_tl)
        v_tr = ap_tr - box_cent[:,ii[0],ii[1],ii[2]]
        v_tr /= np.linalg.norm(v_tr)
        v_bl = ap_bl - box_cent[:,ii[0],ii[1],ii[2]]
        v_bl /= np.linalg.norm(v_bl)
        v_br = ap_br - box_cent[:,ii[0],ii[1],ii[2]]
        v_br /= np.linalg.norm(v_br)

        # Vector angles in binormal direction
        bn_tl = np.dot(
            omega_binorm[:,ii[0],ii[1],ii[2]],
            v_tl - omega_norm[:,ii[0],ii[1],ii[2]]
            )
        bn_tr = np.dot(
            omega_binorm[:,ii[0],ii[1],ii[2]],
            v_tr - omega_norm[:,ii[0],ii[1],ii[2]]
            )
        bn_bl = np.dot(
            omega_binorm[:,ii[0],ii[1],ii[2]],
            v_bl - omega_norm[:,ii[0],ii[1],ii[2]]
            )
        bn_br = np.dot(
            omega_binorm[:,ii[0],ii[1],ii[2]],
            v_br - omega_norm[:,ii[0],ii[1],ii[2]]
            )

        # Vector angles in certical direction
        ve_tl = np.arctan(np.dot(
            omega_vert[:,ii[0],ii[1],ii[2]],
            v_tl - omega_norm[:,ii[0],ii[1],ii[2]]
            ))
        ve_tr = np.arctan(np.dot(
            omega_vert[:,ii[0],ii[1],ii[2]],
            v_tr - omega_norm[:,ii[0],ii[1],ii[2]]
            ))
        ve_bl = np.arctan(np.dot(
            omega_vert[:,ii[0],ii[1],ii[2]],
            v_bl - omega_norm[:,ii[0],ii[1],ii[2]]
            ))
        ve_br = np.arctan(np.dot(
            omega_vert[:,ii[0],ii[1],ii[2]],
            v_br - omega_norm[:,ii[0],ii[1],ii[2]]
            ))

        # Replaces binormal limits
        omega_dl[1,0] = np.min([
            omega_dl[1,0],
            np.min(np.r_[bn_tl, bn_tr, bn_bl, bn_br])
            ])
        omega_dl[1,1] = np.max([
            omega_dl[1,1],
            np.max(np.r_[bn_tl, bn_tr, bn_bl, bn_br])
            ])

        # Replaces vertical limits
        omega_dl[0,0] = np.min([
            omega_dl[0,0],
            np.min(np.r_[ve_tl, ve_tr, ve_bl, ve_br])
            ])
        omega_dl[0,1] = np.max([
            omega_dl[0,1],
            np.max(np.r_[ve_tl, ve_tr, ve_bl, ve_br])
            ])

    if debug:
        dl = 16
        ends = box_cent + dl*omega_norm

        fig, ax = plt.subplots(1,2)

        ax[0].plot(
            np.vstack((box_cent[0,:].flatten(), ends[0,:].flatten())),
            np.vstack((box_cent[1,:].flatten(), ends[1,:].flatten())),
            'b-'
            )

        ax[0].plot([ap_tl[0], ap_tr[0]], [ap_tl[1], ap_tr[1]], 'k-')
        ax[0].plot([ap_tr[0], ap_br[0]], [ap_tr[1], ap_br[1]], 'k-')
        ax[0].plot([ap_tl[0], ap_bl[0]], [ap_tl[1], ap_bl[1]], 'k-')
        ax[0].plot([ap_bl[0], ap_br[0]], [ap_bl[1], ap_br[1]], 'k-')

        ax[0].set_xlabel('X [m]')
        ax[0].set_ylabel('Y [m]')

        ax[1].plot(
            np.vstack((
                np.sqrt(box_cent[0,:].flatten()**2+box_cent[1,:].flatten()**2), 
                np.sqrt(ends[0,:].flatten()**2+ends[1,:].flatten()**2)
                )),
            np.vstack((box_cent[2,:].flatten(), ends[2,:].flatten())),
            'b-'
            )

        ax[1].plot(
            [np.sqrt(ap_tl[0]**2+ap_tl[1]**2), np.sqrt(ap_tr[0]**2+ap_tr[1]**2)],
            [ap_tl[2], ap_tr[2]], 'k-')
        ax[1].plot(
            [np.sqrt(ap_tr[0]**2+ap_tr[1]**2), np.sqrt(ap_br[0]**2+ap_br[1]**2)], 
            [ap_tr[2], ap_br[2]], 'k-')
        ax[1].plot(
            [np.sqrt(ap_tl[0]**2+ap_tl[1]**2), np.sqrt(ap_bl[0]**2+ap_bl[1]**2)],
            [ap_tl[2], ap_bl[2]], 'k-')
        ax[1].plot(
            [np.sqrt(ap_bl[0]**2+ap_bl[1]**2), np.sqrt(ap_br[0]**2+ap_br[1]**2)], 
            [ap_bl[2], ap_br[2]], 'k-')

        ax[1].set_xlabel('R [m]')
        ax[1].set_ylabel('Z [m]')

    # Output
    return omega_norm, omega_vert, omega_binorm, omega_dl


# Discretizes volume-of-sight spatially
def _build_boxes(
    coll = None,
    key_diag = None,
    key_cam  = None,
    lamb0 = None, # [AA]
    # Controls
    dvol = None, 
    dplasma = None,
    ):

    # (R,Z) volume limits, [m], (SPARC specific!!!!)
    # Gets default plasma geometry
    if dplasma is None:
        dplasma = dp.get_dplasma(option='default')
    R0 = dplasma['R0']
    aa = dplasma['aa']

    # Defaults
    if dvol is None:
        dvol = dv.get_dvol(option='default')

    # Finds pt closest to the magnetic axis for this wavelength
    # NOTE: per the midplane LOS
    dlos = utils._get_tofu_los(
        coll = coll,
        key_diag = key_diag,
        key_cam = key_cam,
        lamb0 = lamb0
        )

    # Central box on magnetic axis
    #C0 = tmps[:,ind_r]
    C0 = dlos['mag_axis']
    
    # Vector basis
    #vn = mid_v
    vn = dlos['los_vect']
    vz = np.r_[0,0,1]
    vb = np.cross(vz,vn)

    # Step controls
    nmin = int(-1*dvol['nsteps']['nn'])
    nmax = int(dvol['nsteps']['nn']+1)
    zmin = int(-dvol['nsteps']['nz'])
    zmax = int(dvol['nsteps']['nz']+1)
    bmin = int(-dvol['nsteps']['nb'])
    bmax = int(dvol['nsteps']['nb']+1)

    ln = dvol['lsteps']['ln']
    lz = dvol['lsteps']['lz']
    lb = dvol['lsteps']['lb']

    nn = dvol['nsteps']['nn']
    nz = dvol['nsteps']['nz']
    nb = dvol['nsteps']['nb']

    # Init
    box_cent = np.zeros((3, int(nmax-nmin), int(zmax-zmin), int(bmax-bmin))) # dim(3, nn, nz, nb)

    # Calculates box centers
    for dn in np.arange(nmin,nmax):
        for dz in np.arange(zmin,zmax):
            for db in np.arange(bmin,bmax):

                box_cent[:,nn+dn, nz+dz, nb+db] = (
                    C0
                    + dn *ln *vn
                    + dz *lz *vz
                    + db *lb *vb
                    )

                #Cr = np.sqrt(
                #    box_cent[0,nn+dn, nz+dz, nb+db]**2
                #    + box_cent[1,nn+dn, nz+dz, nb+db]**2
                #    )

                # Figure of merit for box out of bounds
                margin = 0.5
                Cl = (
                    box_cent[:,nn+dn, nz+dz, nb+db]
                    + margin *0.5*ln*vn
                    )
                Clr = np.sqrt(Cl[0]**2+Cl[1]**2)
                Cu = (
                    box_cent[:,nn+dn, nz+dz, nb+db]
                    - margin *0.5*ln*vn
                    )
                Cur = np.sqrt(Cu[0]**2+Cu[1]**2)
                #import pdb; pdb.set_trace()
                if Clr < (R0-aa) or Cur > (R0+aa):
                    box_cent[:,nn+dn, nz+dz, nb+db] *= np.nan

    # Debug plotting
    if dvol['plt']:
        conf = tf.load_config('SPARC-V0')
        poly = conf.Ves.FirstWallV0.Poly

        fig, ax = plt.subplots(1,2)

        ax[0].plot(xx0.flatten()*100, yy0.flatten()*100, 'r*')
        ax[0].plot(losx*100, losy*100,'b-')

        ax[0].plot(box_cent[0].flatten()*100, box_cent[1].flatten()*100, 'g*')

        ax[0].set_xlabel('X [cm]')
        ax[0].set_ylabel('Y [cm]')
        ax[0].grid('on')

        ax[1].plot(rr0.flatten()*100, zz0.flatten()*100, 'r*')
        ax[1].plot(losr*100, losz*100, 'b-')

        ax[1].plot(poly[0]*100, poly[1]*100, 'k-')

        ax[1].plot(np.sqrt(box_cent[0]**2+box_cent[1]**2).flatten()*100, 
            box_cent[2].flatten()*100,
            'g*'
            )    

        ax[1].set_xlabel('R [cm]')
        ax[1].set_ylabel('Z [cm]')
        ax[1].grid('on')

    # Box side lengths, [m]
    box_dl = [ln,lz,lb]

    # Box vector, (norm, vert, binorm)
    box_vect = [vn,vz,vb]

    # Output, dim(3, norm, vert, binorm)
    return box_cent, box_dl, box_vect



#############################################################
#
#               Utilities
#
#############################################################


def _rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    # Error handling if vectors are too close together
    if 1-abs(np.dot(vec1,vec2)) < 1e-10:
        return np.eye(3)

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
