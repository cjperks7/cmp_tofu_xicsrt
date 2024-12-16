'''

Script to run post-processing for emissivity batch runs

cjperks
Dec 16, 2024

'''

# Modules
import numpy as np
import os, sys
import cmp_tofu_xicsrt.utils as utils

# Folder to mv results
fol = os.path.join(
    '/nobackup1/cjperks/work
    'cmp_tofu_xicsrt/HPC',
    'output'
    )

fol_xi = 'XRSHRKr_emis_v2'
fol_tf = 'XRSHRKr_emis_v1'
name = 'SPARC_PRD_XRSHRKr'

# Extracts HPC data
ddata = utils._get_mv_results(
    folder = fol,
    folder_xi = fol_xi,
    folder_tf = fol_tf,
    name = name,
    case = 'me'
    )

# Saves data
np.savez(
    os.path.join(fol, fol_xi, name+'_all_data.npz'),
    ddata
    )

