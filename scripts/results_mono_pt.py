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
lamb0 = 1.61 # [AA]
cry_shape = 'Spherical'
#cry_shape = 'Cylindrical'

# Save controls
dsave = {
    'path': os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt',
        'output',
        'sph_pt_v2'
        ),
    'name': 'pt_sph'
    }


### --- Runs ray-tracing --- ###

# Loads collection object
dout, coll = utils.main(
    lamb0 = lamb0, # [AA]
    coll_tf=os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt/diags',
        'valid_spherical_128x64.npz'
        #'valid_cylindrical_128x64.npz'
        ),
    cry_shape = cry_shape,
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
        'sph_pt_v2',
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
    key_diag = 'valid',
    cry_shape = cry_shape,
    lamb0 = lamb0,
    dout = dout,
    dpt = None,
    )

