'''

Functions to plot comparison between ToFu and XICSRT for
a monoenergetic, volumetric source

cjperks
Aug 26, 2024

'''

# Modules
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    'plt_mono_vol'
    ]

################################################
#
#               Main
#
################################################


# Plots point source results
def plt_mono_vol(
    coll = None,
    key_diag = None,
    cry_shape = 'Spherical',
    lamb0 = None,
    dout = None,
    ):


    # Init
    dxi = dout['XICSRT']
    dtf = dout['ToFu']

    # Rescale ToFu if different pixel binning
    scalet_0 = (
        dtf['npix'][0]/dxi['npix'][0]
        *dtf['npix'][1]/dxi['npix'][1]
        )
    scalet_1 = (
        dtf['npix'][1]/dxi['npix'][1]
        )

    cmax = np.max((
        np.max(dxi['signal'].flatten()),
        np.max(dtf['signal'].flatten())*scalet_0
        ))

    # Plots photon flux on detector from XICSRT
    fig, ax = plt.subplots(1,2)
    im = ax[1].imshow(
        dxi['signal'].T, # normalize [# ph] detected by [# ph] emitted
        extent = dxi['extent'],
        interpolation='nearest',
        origin='lower',
        vmin=0,
        vmax=np.max(dxi['signal'].flatten()),
        aspect = dxi['aspect']
        )
    cb = plt.colorbar(im, ax=ax, orientation='vertical')
    ax[1].set_title('XICSRT, # ph detected = %1.5e'%(
        np.sum(dxi['signal'].flatten())
        ), color = 'blue')
    ax[1].set_xlabel('horizontal bin')


    im1 = ax[0].imshow(
        dtf['signal'].T*scalet_0,
        #extent = extent,
        extent = dtf['extent'],
        interpolation='nearest',
        origin='lower',
        vmin=0,
        vmax=np.max(dtf['signal'].flatten())*scalet_0,
        aspect = dtf['aspect']
        )
    #cb = plt.colorbar(im1, ax=ax1, orientation='horizontal')
    ax[0].set_title('ToFu, # ph detected = %1.5e'%(
        np.sum(dtf['signal'].flatten())
        ), color = 'red')
    ax[0].set_xlabel('horizontal bin')
    ax[0].set_ylabel('vertical bin')
    cb.set_label('#ph/bin^2')


    fig.show()

