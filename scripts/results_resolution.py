'''

Script to plot spatial-/spectral-resolution results

cjperks
Nov 9. 2024

'''

# Modules
import sys, os

from cmp_tofu_xicsrt import plotting as pl

# Enables automatic reloading of modules
%reload_ext autoreload
%autoreload 2

# File management
dout = os.path.join(
    '/home/cjperks',
    'cmp_tofu_xicsrt/output',
    'XRSHRKr_res_v2',
    'res_XRSHRKr_lamb0.94500AA.npz'
    )

# Plotting
pl.plt_res(
    dout=dout
    )


