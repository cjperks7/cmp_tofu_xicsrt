'''

Function to build diagnostic as a ToFu Collection object

cjperks
Aug 5, 2024

'''

# Modules
import tofu as tf

import numpy as np

import cmp_tofu_xicsrt.setup._def_diag as dd
import cmp_tofu_xicsrt.utils as utils

__all__ = [
    '_init_diag'
    ]

#################################################
#
#               Main
#
#################################################

# Load diagnostic from ToFu
def _init_diag(
    coll = None,
    dplasma = None,
    dcry = None,
    lamb0 = None,
    subcam = None,
    doptics = None,
    ):

    # If user is reloading a Collection object
    if isinstance(coll, str):
        coll = tf.data.load(coll)

    # Else, build a new diagnostic
    else:
        coll = _build_diag(
            dcry = dcry,
            lamb0 = lamb0,
            subcam = subcam,
            doptics = doptics,
            )

        # Saves diagnostic
        coll.save(path='/home/cjperks/cmp_tofu_xicsrt/diags')

    # Output
    return coll


#################################################
#
#               Utilities
#
#################################################

# Script to build diagnostic in ToFu
def _build_diag(
    dap = None,
    dcry = None,
    dmat = None,
    dcam = None,
    lamb0 = None,
    subcam = None,
    doptics = None,
    ):

    # Labeling
    if doptics is not None:
        ap_option = doptics['key_diag']
        ap_label = doptics['key_ap']

        cry_option = doptics['key_diag']
        mat_option = doptics['key_diag']
        cry_label = doptics['key_cry']

        cam_option = doptics['key_cam']
        cam_label = doptics['key_cam']

        diag_label = doptics['key_diag']

    else:
        ap_option = 'default'
        ap_label = 'ap'

        cry_option = dcry
        mat_option = 'default'
        cry_label = 'cry'

        cam_options = 'default'
        cam_label = 'cam'

        diag_label = 'valid'

    # Init
    coll = tf.data.Collection()

    # Gets default aperture geometry
    if dap is None:
        dap = dd.get_dap(option=ap_option)

    # Adds aperture
    coll.add_aperture(
        key=ap_label,
        **dap
        )

    # Gets dafault crystal geometry
    if isinstance(cry_option, str):
        dcry = dd.get_cry_dgeom(option=cry_option)
    if dmat is None:
        dmat = dd.get_cry_dmat(option=mat_option)
    dmat['target']['lamb'] = lamb0*1e-10

    # Adds crystal
    coll.add_crystal(
        key = cry_label,
        dgeom = dcry,
        dmat = dmat,
        )

    # Gets default camera geometry
    if dcam is None:
        dcam = dd.get_dcam(option=cam_option)

    # Adds camera
    coll.add_camera_2d(
        key = cam_label,
        dgeom = dcam 
        )

    # Takes a subset of the camera area if requested
    if subcam is not None:
        dsub = dd.get_dsubcam(option=subcam)

        # If user wants to look at a section of the detector in a finer pixel mesh
        if dsub['method'] == 'refine':
            if dsub['dx'] is not None:
                dcam['cent'] += dcam['e0']*dsub['dx']
                nx0 = len(dcam['cents_x0'])
                pix_width = 2*dsub['xhalfsize']/(nx0-1)

                dcam['outline_x0'] = 0.5* pix_width * np.r_[-1, 1, 1, -1]
                dcam['cents_x0'] = dsub['xhalfsize'] * np.linspace(-1, 1, nx0)
            if dsub['dy'] is not None:
                dcam['cent'] += dcam['e1']*dsub['dy']
                nx1 = len(dcam['cents_x1'])
                pix_height = 2*dsub['yhalfsize']/(nx1-1)

                dcam['outline_x1'] = 0.5* pix_height * np.r_[-1, -1, 1, 1]
                dcam['cents_x1'] = dsub['yhalfsize'] * np.linspace(-1, 1, nx1)

        # If user wants to zoom into a section of the detector on the same pixel mesh
        elif dsub['method'] == 'zoom':
            if dsub['dx'] is not None:
                indx_up = np.where(
                    (
                        coll.ddata[cam_label+'_c0']['data'] 
                        + coll.dobj['camera'][cam_label]['dgeom']['extenthalf'][0]
                        ) <= dsub['dx'] + dsub['xhalfsize']
                    )[0][-1] # Last pixel physically before cutoff
                indx_low = np.where(
                    (
                        coll.ddata[cam_label+'_c0']['data'] 
                        + coll.dobj['camera'][cam_label]['dgeom']['extenthalf'][0]
                        ) <= dsub['dx'] - dsub['xhalfsize']
                    )[0][-1] # Last pixel physically before cutoff

                dcam['cents_x0'] = (
                    coll.ddata[cam_label+'_c0']['data'][indx_low:indx_up]
                    )

            if dsub['dy'] is not None:
                indy_up = np.where(
                    (
                        coll.ddata[cam_label+'_c1']['data'] 
                        + coll.dobj['camera'][cam_label]['dgeom']['extenthalf'][1]
                        ) <= dsub['dy'] + dsub['yhalfsize']
                    )[0][-1] # Last pixel physically before cutoff
                indy_low = np.where(
                    (
                        coll.ddata[cam_label+'_c1']['data'] 
                        + coll.dobj['camera'][cam_label]['dgeom']['extenthalf'][1]
                        ) <= dsub['dy'] - dsub['yhalfsize']
                    )[0][-1] # Last pixel physically before cutoff

                dcam['cents_x1'] = (
                    coll.ddata[cam_label+'_c1']['data'][indy_low:indy_up]
                    )

        # Adds subcamera
        cam_label += '_' + diag_label
        coll.add_camera_2d(
            key = cam_label,
            dgeom = dcam 
            )

    # Builds diagnostic
    coll.add_diagnostic(
        key = diag_label,
        doptics = {cam_label: [cry_label, ap_label]},
        compute = True, # compute LOS
        compute_vos_from_los = False,
        convex = True,
        config = tf.load_config('SPARC-V0')
        )

    # Output
    return coll











