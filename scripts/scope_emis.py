'''

Script to scope local emissivity data within the VOS 
of a diagnostic projected in the poloidal cross-section

cjperks
Nov 7th, 2024

'''

# Modules
import sys, os

from cmp_tofu_xicsrt.plotting import _scope_emis as se

# Enables automatic reloading of modules
%reload_ext autoreload
%autoreload 2

# File management
emis_file = os.path.join(
    '/home/cjperks/',
    'cmp_tofu_xicsrt/emis',
    'XRSHRKr.npz'
    ) # He-like Kr, PRD profiles
gfile = os.path.join(
    '/home/cjperks',
    'tofu_sparc/background_plasma',
    'PRD_plasma/run1',
    'input.geq'
    ) # PRD plasma
coll_tf = os.path.join(
    '/home/cjperks',
    'cmp_tofu_xicsrt/diags',
    'sparc_htpd24_v2.npz'
    )

# Plotting
se.scope_emis(
    coll_tf = coll_tf,
    key_diag = 'XRSHRKr',
    key_cam = 'e1M3_XRSHRKr',
    emis_file = emis_file,
    gfile = gfile,
    lamb0 = 0.945,
    )