'''

Functions to plot comparison between ToFu and XICSRT for
a monoenergetic, point source

cjperks
Aug 5, 2024

'''

# Modules
import numpy as np
import matplotlib.pyplot as plt

import cmp_tofu_xicsrt.setup._def_point as dp

__all__ = [
    'plt_mono_pt'
    ]

################################################
#
#               Main
#
################################################


# Plots point source results
def plt_mono_pt(
    coll = None,
    key_diag = None,
    lamb0 = None,
    dout = None,
    dpt = None,
    ):

    # Gets default values
    if dpt is None:
        dpt = dp.get_dpt(option='default')

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






    

    fig.set_size_inches(20, 8)

    # If no user defined horizontal bin, use one that's maximum
    if dpt['plotting']['xind'] is None:
        xind = np.argmax(
            np.sum(dxi['signal'], axis=1)
            )
    else:
        xind = dpt['plotting']['xind']

    xind_tf = np.argmin(abs(dxi['cents_cm'][0][xind]-dtf['cents_cm'][0]))


    # Comparison plot of photon flux at fixed Horizontal bin
    plt.rcParams.update({'font.size': 18})
    fig2,ax2=plt.subplots(1,2)
    lw = 3
    ms = 10
    pa = 20

    ax2[0].plot(
        dxi['cents_cm'][1],
        dxi['signal'][xind,:],
        'b*-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax2[0].plot(
        dtf['cents_cm'][1],
        dtf['signal'][xind_tf,:]*scalet_0,
        'r*-',
        label = 'ToFu',
        linewidth = lw,
        markersize = ms
        )

    ax2[0].grid('on')
    ax2[0].set_xlabel('vertical bin')
    ax2[0].set_ylabel('# photons/bin^2')
    leg = ax2[0].legend()
    leg.set_draggable('on')
    ax2[0].set_title('horiz. bin %0.0i/%0.0i'%(
            xind, dxi['npix'][0]-1
            ),
        pad = pa
        )

    ax2[1].plot(
        dxi['cents_cm'][1],
        np.sum(dxi['signal'],axis=0),
        'b*-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax2[1].plot(
        dtf['cents_cm'][1],
        np.sum(dtf['signal'],axis=0)*scalet_1,
        'r*-',
        label = 'ToFu',
        linewidth = lw,
        markersize = ms
        )
    ax2[1].set_title('int. over all horiz. bin', pad = pa)
    ax2[1].grid('on')
    ax2[1].set_xlabel('vertical bin')
    ax2[1].set_ylabel('# photons/bin')
    
    #ax2.set_ylim(0, 5e-12)
    fig2.suptitle('Detector binned (%0.0i, %0.0i), point = [%1.2f, %1.2f, %1.2f] m'%(
            dxi['npix'][0]-1, dxi['npix'][1]-1, 
            #lamb0, 
            dpt['ToFu']['point'][0], dpt['ToFu']['point'][1], dpt['ToFu']['point'][2]
            )
        )

    ymax = np.max((
        np.max(np.sum(dxi['signal'], axis=0)),
        np.max(np.sum(dtf['signal'],axis=0))*scalet_1
        ))
    ax2[0].set_ylim(0,1.1*ymax)
    ax2[1].set_ylim(0,1.1*ymax)

    fig2.set_size_inches(10,8)

    # Integrated photons on detector
    print('Normalized # ph detected')
    print('ToFu')
    print(np.sum(dtf['signal'].flatten()))
    print('XICSRT')
    print(np.sum(dxi['signal'].flatten()))


 
