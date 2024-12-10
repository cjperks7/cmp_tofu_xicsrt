'''

Script to run radially-peaked SPARC emissivity profile
using X-ray diagnostics per HTPD 2024:
C. Perks, Rev. Sci. Instrum. 95, 083555 (2024)

cjperks
Sep 9th , 2024

'''

# Modules
import sys, os
import time
import cmp_tofu_xicsrt.utils as utils

start_time = time.time()

# Diagnostic controls
key_diag = 'XRSHRKr'

cry_shape = 'Cylindrical'
key_cam = 'e1M3_'+key_diag
emis_file = os.path.join(
    '/home/cjperks',
    'cmp_tofu_xicsrt/emis',
    key_diag+'.npz'
    )

# HPC controls
lamb_lim = 100 # Maximum number of wavelengths in the mesh to run in one job
nlamb = 600
njob_lamb = int(nlamb/lamb_lim)

job_max_vol = 34+1

dHPC = {
    'job_axis': ['vert' ,'binorm'],
    'job_num': int(sys.argv[1])%(job_max_vol),    # Which volume group
    'job_max': int(job_max_vol-1), # Pythonic, 0,...,jm
    'num_rays': 1e5,# (20G) 1e8 (100G)
    'nlamb': nlamb,
    'lamb_lim': lamb_lim,
    'lamb_num': int(int(sys.argv[1])/job_max_vol),   # Which wavelength group
    }

# Save controls
dsave = {
    'path': os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt',
        'HPC/output',
        sys.argv[2]
        ),
    'name': 'SPARC_PRD'
    }


if sys.argv[3] == 'tofu':
    run_tofu = True
    run_xicsrt = False
elif sys.argv[3] == 'xicsrt':
    run_tofu = False
    run_xicsrt = True


# Loads collection object
dout, coll = utils.main(
    coll_tf=os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt/diags',
        'sparc_htpd24_v2.npz'
        ),
    emis_file = emis_file,
    key_diag = key_diag,
    key_cam = key_cam,
    cry_shape = cry_shape,
    # Multi-energy, volumetric source controls
    rad_run = True,
    dvol = None,
    # HPC controls
    run_tofu = run_tofu,
    run_xicsrt = run_xicsrt,
    dHPC = dHPC,
    dsave = dsave,
    )

end_time = time.time()
dt = end_time - start_time
print(f"Elapsed time: {dt} seconds")