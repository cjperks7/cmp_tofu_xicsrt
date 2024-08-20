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
#cry_shape = 'Spherical'
cry_shape = 'Cylindrical'

# Save controls
dsave = {
    'path': os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt',
        'output',
        'pt_v1'
        ),
    'name': 'pt_cyl'
    }


### --- Runs ray-tracing --- ###

# Loads collection object
dout, coll = utils.main(
    lamb0 = lamb0, # [AA]
    coll_tf=os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt/diags',
        #'valid_spherical_128x64.npz'
        'valid_cylindrical_128x64.npz'
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



# Plots results
plotting.plt_mono_pt(
    coll = coll,
    key_diag = 'valid',
    cry_shape = cry_shape,
    lamb0 = lamb0,
    dout = dout,
    dpt = None,
    )

