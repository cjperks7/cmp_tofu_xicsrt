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
    ddata = None,
    ):

    ###########
    # --- Detector images
    ##########


    # Init
    dxi = ddata['XICSRT']
    dtf = ddata['ToFu']

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

    print('Total Flux Error:')
    print('%0.2f %%'%(
        (1-np.sum(dxi['signal'].flatten())/np.sum(dtf['signal'].flatten()))*100
        )) #

    ###########
    # --- Image Slices
    ##########

    indy = int(dxi['npix'][1]/2-1)
    indx = np.argmax(np.sum(dxi['signal'], axis=1))

    indyt = int(dtf['npix'][1]/2-1)
    indxt = np.argmin(abs(
        dtf['cents_cm'][0] - dxi['cents_cm'][0][indx]
        ))

    # Rescale ToFu if different pixel binning
    scalet_10 = scalet_11 = (
        dtf['npix'][0]/dxi['npix'][0]
        *dtf['npix'][1]/dxi['npix'][1]
        )
    scalet_01 = (
        dtf['npix'][1]/dxi['npix'][1]
        )
    scalet_00 = (
        dtf['npix'][0]/dxi['npix'][0]
        )

    fig4, ax4 = plt.subplots(2,2)


    ax4[0,0].plot(
        dxi['cents_cm'][0],
        np.sum(dxi['signal'], axis = 1),
        'b*-',
        label = 'XICSRT'
        )

    ax4[0,0].plot(
        dtf['cents_cm'][0],
        np.sum(dtf['signal'], axis = 1) *scalet_00,
        'r*-',
        label = 'ToFu'
        )

    ax4[0,0].set_xlabel('horiz. bin [cm]')
    ax4[0,0].set_ylabel('# photons/bin')
    ax4[0,0].set_title('int. over all vert. bins')
    ax4[0,0].grid('on')
    ax4[0,0].legend(labelcolor='linecolor')

    ax4[1,0].plot(
        dxi['cents_cm'][0],
        dxi['signal'][:,indy],
        'b*-'
        )

    ax4[1,0].plot(
        dtf['cents_cm'][0],
        dtf['signal'][:,indyt] *scalet_10,
        'r*-'
        )

    ax4[1,0].set_xlabel('horiz. bin [cm]')
    ax4[1,0].set_ylabel('# photons/bin^2')
    ax4[1,0].set_title('vert. bin %i/%i'%(indy, dxi['npix'][1]-1))
    ax4[1,0].grid('on')

    ax4[0,1].plot(
        dxi['cents_cm'][1],
        np.sum(dxi['signal'], axis = 0),
        'b*-'
        )

    ax4[0,1].plot(
        dtf['cents_cm'][1],
        np.sum(dtf['signal'], axis = 0) *scalet_01,
        'r*-'
        )

    ax4[0,1].set_xlabel('vert. bin [cm]')
    ax4[0,1].set_ylabel('# photons/bin')
    ax4[0,1].set_title('int. over all horz. bins')
    ax4[0,1].grid('on')

    ax4[1,1].plot(
        dxi['cents_cm'][1],
        dxi['signal'][indx,:],
        'b*-'
        )

    ax4[1,1].plot(
        dtf['cents_cm'][1],
        dtf['signal'][indxt,:] *scalet_11,
        'r*-'
        )


    ax4[1,1].set_xlabel('vert. bin [cm]')
    ax4[1,1].set_ylabel('# photons/bin^2')
    ax4[1,1].set_title('horz. bin %i/%i'%(indx, dxi['npix'][0]-1))
    ax4[1,1].grid('on')










