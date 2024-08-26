'''

Script to read results of a mono-energetci, volumetric source simulation

cjperks
Aug 26, 2024

'''

# Modules
import numpy as np
import os
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
plotting.plt_mono_pt(
    dout = ddata,
    )


