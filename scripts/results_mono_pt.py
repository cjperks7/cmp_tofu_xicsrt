'''

Script to compare results between ToFu and XICSRT for 
a monoenergetic, point source

cjperks
Aug 5, 2024

'''

# Modules
import cmp_tofu_xicsrt.utils as utils
import cmp_tofu_xicsrt.plotting as plotting
import os

# Enables automatic reloading of modules
%reload_ext autoreload
%autoreload 2

### --- Controls --- ###

# Wavelength
#lamb0 = 1.61 # [AA]
lamb0 = 0.945 # [AA]
#cry_shape = 'Spherical'
cry_shape = 'Cylindrical'

key_diag = 'XRSHRKr'
key_cam = 'e1M3_XRSHRKr'

# Save controls
dsave = {
    'path': os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt',
        'output',
        #'sph_pt_v3'
        'XRSHRKr_pt_v1'
        ),
    #'name': 'pt_sph'
    'name': 'pt_XRSHRKr'
    }


### --- Runs ray-tracing --- ###

# Loads collection object
dout, coll = utils.main(
    lamb0 = lamb0, # [AA]
    coll_tf=os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt/diags',
        'sparc_htpd24_v2.npz'
        #'valid_spherical_128x64.npz'
        #'valid_cylindrical_128x64.npz'
        ),
    cry_shape = cry_shape,
    key_diag = key_diag,
    key_cam = key_cam,
    # Monochromatic, point source controls
    pt_run = True,
    dpt = None,
    # HPC controls
    run_tofu =True,
    run_xicsrt = True,
    dsave = dsave,
    niter = 5,
    )



'''
import cmp_tofu_xicsrt.plotting as plotting
import tofu as tf
import os

# Enables automatic reloading of modules
%reload_ext autoreload
%autoreload 2


coll = tf.data.load(
    os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt/diags',
        'valid_spherical_128x64.npz'
        #'valid_cylindrical_128x64_v2.npz'
        )
    )
dout = np.load(
    os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt/output',
        #'cyl_pt_v1',
        #'pt_cyl_lamb1.61000AA.npz'
        'sph_pt_v3',
        'pt_sph_lamb1.61000AA.npz'
        ),
    allow_pickle=True
    )['arr_0'][()]

#cry_shape = 'Cylindrical'
cry_shape = 'Spherical'
lamb0 = 1.61

'''


# Plots results
plotting.plt_mono_pt(
    coll = coll,
    key_diag = key_diag,
    cry_shape = cry_shape,
    lamb0 = lamb0,
    dout = dout,
    dpt = None,
    plt_rc = False,
    )

