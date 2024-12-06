'''

Functions to plot comparison between ToFu and XICSRT for
a monoenergetic, point source

cjperks
Aug 5, 2024

'''

# Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 16})

import cmp_tofu_xicsrt.setup._def_point as dp
import cmp_tofu_xicsrt.utils as utils

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
    cry_shape = 'Spherical',
    lamb0 = None, # [AA]
    dout = None,
    dpt = None,
    plt_rc = False,
    ):

    # Gets default values
    if dpt is None:
        dpt = dp.get_dpt(option=cry_shape)

    # Init
    dxi = dout['XICSRT']
    dtf = dout['ToFu']

    dx_xi = np.mean(abs(dxi['cents_cm'][0][1:] - dxi['cents_cm'][0][:-1]))
    dy_xi = np.mean(abs(dxi['cents_cm'][1][1:] - dxi['cents_cm'][1][:-1]))
    dx_tf = np.mean(abs(dtf['cents_cm'][0][1:] - dtf['cents_cm'][0][:-1]))
    dy_tf = np.mean(abs(dtf['cents_cm'][1][1:] - dtf['cents_cm'][1][:-1]))

    # Rescale ToFu if different pixel binning
    #scalet_0 = (
    #    dtf['npix'][0]/dxi['npix'][0]
    #    *dtf['npix'][1]/dxi['npix'][1]
    #    )
    #scalet_1 = (
    #    dtf['npix'][1]/dxi['npix'][1]
    #    )
    scalet_0 = (
        dx_tf/dx_xi
        *dy_tf/dy_tf
        )
    scalet_1 = (
        dy_tf/dy_xi
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
        np.sum(dtf['signal'].flatten())*scalet_0
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
    #xind_tf = np.argmax(np.sum(dtf['signal'], axis=1))


    # Comparison plot of photon flux at fixed Horizontal bin
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
    print(np.sum(dtf['signal'].flatten())*scalet_0)
    print('XICSRT')
    print(np.sum(dxi['signal'].flatten()))


    #####################################################################
    #
    #           Used in paper
    #
    ####################################################################

    ###### ---- Detector image ---- ######

    fig3 = plt.figure(figsize = (16,10))
    gs3 = gridspec.GridSpec(2,2, width_ratios = [2,1], hspace = 0.30)
    ms = 8

    ##### PLots XICSRT detector image ######
    ax30 = fig3.add_subplot(gs3[1,0])

    im = ax30.imshow(
        dxi['signal'].T, # normalize [# ph] detected by [# ph] emitted
        extent = np.asarray(dxi['extent'])*100,
        interpolation='nearest',
        origin='lower',
        vmin=0,
        vmax=np.max(dxi['signal'].flatten()),
        aspect = 'equal'
        )

    ax30.set_title('XICSRT, #ph detected = %1.3e'%(
        np.sum(dxi['signal'].flatten())
        ), color = 'blue')
    ax30.set_xlabel('horiz. bin [cm]')
    ax30.set_ylabel('vert. bin [cm]')

    cb = fig3.colorbar(im, ax=ax30, orientation='vertical')
    cb.set_label(r'signal [#ph/bin$^2$]')

    ax30.text(0.05, 0.90, '(b)', color = 'w', transform=ax30.transAxes)


    ##### PLots ToFu detector image ######
    ax31 = fig3.add_subplot(gs3[0,0])


    im1 = ax31.imshow(
        dtf['signal'].T*scalet_0,
        #extent = extent,
        extent = np.asarray(dtf['extent'])*100,
        interpolation='nearest',
        origin='lower',
        vmin=0,
        vmax=np.max(dtf['signal'].flatten())*scalet_0,
        aspect = 'equal'
        )
    ax31.set_title('ToFu, #ph detected = %1.3e'%(
        np.sum(dtf['signal'].flatten())
        ), color = 'red')
    ax31.set_xlabel('horiz. bin [cm]')
    ax31.set_ylabel('vert. bin [cm]')

    #cb = fig3.colorbar(im, ax=[ax30, ax31], orientation='vertical')
    cb = fig3.colorbar(im, ax=ax31, orientation='vertical')
    cb.set_label(r'signal [#ph/bin$^2$]')

    ax31.text(0.05, 0.90, '(a)', color = 'w', transform=ax31.transAxes)

    ##### Plots image slice ######
    ax3 = fig3.add_subplot(gs3[0,1])

    ax3.plot(
        #dxi['cents_cm'][1],
        #dxi['signal'][xind,:],
        dxi['signal'][xind,:]*1e12,
        dxi['cents_cm'][1],
        'bD-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        #dtf['cents_cm'][1],
        #dtf['signal'][xind_tf,:]*scalet_0,
        dtf['signal'][xind_tf,:]*scalet_0*1e12,
        dtf['cents_cm'][1],
        'ro-',
        label = 'ToFu',
        linewidth = lw,
        markersize = ms
        )

    ax3.grid('on')
    #ax3.set_xlabel('vert. bin [cm]')
    #ax3.set_ylabel(r'signal [#ph/bin$^2$]')
    ax3.set_ylabel('vert. bin [cm]')
    ax3.set_xlabel(r'signal [1e-12 #ph/bin$^2$]')

    leg = ax3.legend(labelcolor='linecolor')
    leg.set_draggable('on')
    ax3.set_title('horiz. bin %0.0i/%0.0i'%(
            xind+1, dxi['npix'][0]
            ),
        pad = 0
        )

    #ax3.set_ylim(0,1.1*ymax)
    ax3.set_xlim(0,1.1*ymax*1e12)

    ax3.text(0.05, 0.90, '(c)', color = 'k', transform=ax3.transAxes)


    ##### Plots integrated image ######
    ax3 = fig3.add_subplot(gs3[1,1])

    ax3.plot(
        #dxi['cents_cm'][1],
        #np.sum(dxi['signal'],axis=0),
        np.sum(dxi['signal'],axis=0)*1e12,
        dxi['cents_cm'][1],
        'bD-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        #dtf['cents_cm'][1],
        #np.sum(dtf['signal'],axis=0)*scalet_1,
        np.sum(dtf['signal'],axis=0)*scalet_1*1e12,
        dtf['cents_cm'][1],
        'ro-',
        label = 'ToFu',
        linewidth = lw,
        markersize = ms
        )
    ax3.set_title('int. over all horiz. bin', pad = 0)
    ax3.grid('on')
    #ax3.set_xlabel('vert. bin [cm]')
    #ax3.set_ylabel('signal [#ph/bin]')
    ax3.set_ylabel('vert. bin [cm]')
    ax3.set_xlabel('signal [1e-12 #ph/bin]')
    
    #ax3.set_ylim(0,1.1*ymax)
    ax3.set_xlim(0,1.1*ymax*1e12)

    ax3.text(0.05, 0.90, '(d)', color = 'k', transform=ax3.transAxes)

    #fig2.suptitle('Detector binned (%0.0i, %0.0i), point = [%1.2f, %1.2f, %1.2f] m'%(
    #        dxi['npix'][0]-1, dxi['npix'][1]-1, 
    #        #lamb0, 
    #        dpt['ToFu']['point'][0], dpt['ToFu']['point'][1], dpt['ToFu']['point'][2]
    #        )
    #    )


    ###### ---- Rocking curve distributions ---- ######
    if plt_rc:

        # Get XICSRT distributions
        dang_xi = utils._calc_ang_hist_xicsrt(
            data = dout,
            sim_type = 'pt',
            plt_all = False,
            config = dout['XICSRT']['config'],
            nbins = 100,
            )

        # Gets ToFu distributions
        dang_tf = utils._calc_ang_hist_tofu(
            dtf = dtf,
            lamb0 = lamb0*1e-10,
            )

        figr, axr = plt.subplots(1,2, figsize = (8,6))

        axr[0].bar(
            (dang_xi['ang_in'][1:]+dang_xi['ang_in'][:-1])/2*3.6e3,
            dang_xi['hist_in']/np.max(dang_xi['hist_in']),
            width = np.diff(dang_xi['ang_in'])*3.6e3,
            color = 'b'
            )
        axr[0].plot(
            dang_xi['rc_ang']*3.6e3,
            dang_xi['rc_pwr']/np.max(dang_xi['rc_pwr']),
            linewidth = 3,
            color = 'k'
            )

        axr[0].set_xlabel(r'$\theta_{in}-\theta_B$ [arcsec]')
        axr[0].set_ylabel('Reflected power')
        axr[0].grid('on')
        axr[0].set_title('XICSRT', color = 'b')

        #axr[0].set_xlim(dang_xi['ang_in'][0], dang_xi['ang_in'][-1])
        axr[0].set_xlim(-36, 36)
        axr[0].set_ylim(0, 1.05)

        axr[1].plot(
            (dang_tf['ang_in']-dang_xi['bragg'])*3.6e3,
            dang_tf['pwr_in']/np.max(dang_tf['pwr_in']),
            '*',
            color = 'r'
            )
        axr[1].plot(
            dang_xi['rc_ang']*3.6e3,
            dang_xi['rc_pwr']/np.max(dang_xi['rc_pwr']),
            linewidth = 3,
            color = 'k'
            )

        axr[1].set_xlabel(r'$\theta_{in}-\theta_B$ [arcsec]')
        #axr[1].set_ylabel('norm. dist')
        axr[1].grid('on')
        axr[1].set_title('ToFu', color = 'r')

        #axr[1].set_xlim(dang_xi['ang_in'][0], dang_xi['ang_in'][-1])
        axr[1].set_xlim(-36, 36)
        axr[1].set_ylim(0,1.05)



 
