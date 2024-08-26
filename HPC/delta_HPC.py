#!/usr/bin/env python
'''

Script to simulate a volumetric, monoenergetic (Delta) source

Meant to be used on an HPC environment

cjperks
Apr 4, 2024

'''

# Modules
import sys, os
import time
import cmp_tofu_xicsrt.utils as utils

start_time = time.time()

# Wavelength
lamb0 = 1.61 # [AA]
#cry_shape = 'Spherical'
cry_shape = 'Cylindrical'

# HPC controls
dHPC = {
    'job_axis': ['vert' ,'binorm'],
    'job_num': int(sys.argv[1]),
    'job_max': 34, # Pythonic, 0,...,jm
    #'job_axis': ['vert'],
    #'job_num': int(sys.argv[1]),
    #'job_max': 12,
    'num_rays': 1e5# (20G) 1e8 (100G)
    }

# Save controls
dsave = {
    'path': os.path.join(
        '/home/cjperks',
        'cmp_tofu_xicsrt',
        'HPC/output',
        sys.argv[2]
        ),
    'name': 'delta_cyl'
    }
'''
# Prints console stdout to file
sys.stdout = open(
    os.path.join(
        '/home/cjperks',
        'tofu_sparc/ValidationStudy',
        'output',
        sys.argv[2],
        'slurm_%i.out'%(float(sys.argv[1]))
        ),
    'w'
    )
'''

if sys.argv[3] == 'tofu':
    run_tofu = True
    run_xicsrt = False
elif sys.argv[3] == 'xicsrt':
    run_tofu = False
    run_xicsrt = True

# Loads collection object
dout, coll = utils.main(
    lamb0 = lamb0, # [AA]
    #coll_tf=os.path.join(
    #    '/home/cjperks',
    #    'cmp_tofu_xicsrt/diags',
    #    #'valid_spherical_128x64.npz'
    #    'valid_cylindrical_128x64.npz'
    #    ),
    cry_shape = cry_shape,
    # Monochromatic, volumetric source controls
    vol_run = True,
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

#dout = np.load('test_lamb1.61000AA_job32.npz', allow_pickle=True)['arr_0'][()]

