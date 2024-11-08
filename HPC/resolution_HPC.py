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
#%reload_ext autoreload
#%autoreload 2

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
        '/nobackup1/cjperks',
        'work/cmp_tofu_xicsrt',
        'HPC/output',
        sys.argv[1]
        #'XRSHRKr_res_v1'
        ),
    'name': 'res_XRSHRKr'
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
    # Spatial-/spectral-resolution controls
    res_run = True,
    dpt = None,
    dres = None,
    # HPC controls
    run_tofu =True,
    run_xicsrt = True,
    dsave = dsave,
    niter = 5,
    )

