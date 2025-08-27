'''

Functions to plot comparison between ToFu and XICSRT for
a monoenergetic, volumetric source

cjperks
Aug 26, 2024

'''

# Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 16})


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

    # Normalizing by bin length
    scalex = {}
    scalex['h'] = (
        1
        /np.mean(np.diff(dxi['cents_cm'][0]))
        ) # [1/cm], horizontal bin width
    scalex['v'] = (
        1
        /np.mean(np.diff(dxi['cents_cm'][1]))
        ) # [1/cm], vertical bin width
    scalex['hv'] = scalex['h']*scalex['v'] # [1/cm^2], bin area

    scalet = {}
    scalet['h'] = (
        1
        /np.mean(np.diff(dtf['cents_cm'][0]))
        ) # [1/cm], horizontal bin width
    scalet['v'] = (
        1
        /np.mean(np.diff(dtf['cents_cm'][1]))
        ) # [1/cm], vertical bin width
    scalet['hv'] = scalet['h']*scalet['v'] # [1/cm^2], bin area

    label_hv = r'#$ph/cm^2$'
    label_h = label_v = r'#$ph/cm$'

    # Plots photon flux on detector from XICSRT
    fig, ax = plt.subplots(1,2)
    im = ax[1].imshow(
        dxi['signal'].T*scalex['hv'], # normalize [# ph] detected by [# ph] emitted
        extent = dxi['extent'],
        interpolation='nearest',
        origin='lower',
        vmin=0,
        vmax=np.max(dxi['signal'].flatten())*scalex['hv'],
        aspect = dxi['aspect']
        )
    cb = plt.colorbar(im, ax=ax, orientation='vertical')
    ax[1].set_title('XICSRT, # ph detected = %1.5e'%(
        np.sum(dxi['signal'].flatten())
        ), color = 'blue')
    ax[1].set_xlabel('horizontal bin')


    im1 = ax[0].imshow(
        dtf['signal'].T*scalet['hv'],
        #extent = extent,
        extent = dtf['extent'],
        interpolation='nearest',
        origin='lower',
        vmin=0,
        vmax=np.max(dtf['signal'].flatten())*scalet['hv'],
        aspect = dtf['aspect']
        )
    #cb = plt.colorbar(im1, ax=ax1, orientation='horizontal')
    ax[0].set_title('ToFu, # ph detected = %1.5e'%(
        np.sum(dtf['signal'].flatten())
        ), color = 'red')
    ax[0].set_xlabel('horizontal bin')
    ax[0].set_ylabel('vertical bin')
    cb.set_label(label_hv)


    fig.show()

    print('Total Flux Error:')
    print('%0.2f %%'%(
        (1-np.sum(dxi['signal'].flatten())/np.sum(dtf['signal'].flatten()))*100
        )) #

    ###########
    # --- Image Slices
    ##########

    indyt = int(dtf['npix'][1]/2-1)
    indxt = np.argmax(np.sum(dtf['signal'], axis=1))

    indy = np.argmin(abs(
        dtf['cents_cm'][1][indyt] - dxi['cents_cm'][1]
        ))
    indx = np.argmin(abs(
        dtf['cents_cm'][0][indxt] - dxi['cents_cm'][0]
        ))


    fig4, ax4 = plt.subplots(2,2)


    ax4[0,0].plot(
        dxi['cents_cm'][0],
        np.sum(dxi['signal'], axis = 1)*scalex['h'],
        'b*-',
        label = 'XICSRT'
        )

    ax4[0,0].plot(
        dtf['cents_cm'][0],
        np.sum(dtf['signal'], axis = 1)*scalet['h'],
        'r*-',
        label = 'ToFu'
        )

    ax4[0,0].set_xlabel('horiz. bin [cm]')
    ax4[0,0].set_ylabel(label_h)
    ax4[0,0].set_title('int. over all vert. bins')
    ax4[0,0].grid('on')
    ax4[0,0].legend(labelcolor='linecolor')

    ax4[1,0].plot(
        dxi['cents_cm'][0],
        dxi['signal'][:,indy]*scalex['hv'],
        'b*-'
        )

    ax4[1,0].plot(
        dtf['cents_cm'][0],
        dtf['signal'][:,indyt] *scalet['hv'],
        'r*-'
        )

    ax4[1,0].set_xlabel('horiz. bin [cm]')
    ax4[1,0].set_ylabel(label_hv)
    ax4[1,0].set_title('vert. bin %i/%i'%(indy, dxi['npix'][1]-1))
    ax4[1,0].grid('on')

    ax4[0,1].plot(
        dxi['cents_cm'][1],
        np.sum(dxi['signal'], axis = 0)*scalex['v'],
        'b*-'
        )

    ax4[0,1].plot(
        dtf['cents_cm'][1],
        np.sum(dtf['signal'], axis = 0) *scalet['v'],
        'r*-'
        )

    ax4[0,1].set_xlabel('vert. bin [cm]')
    ax4[0,1].set_ylabel(label_v)
    ax4[0,1].set_title('int. over all horz. bins')
    ax4[0,1].grid('on')

    ax4[1,1].plot(
        dxi['cents_cm'][1],
        dxi['signal'][indx,:]*scalex['hv'],
        'b*-'
        )

    ax4[1,1].plot(
        dtf['cents_cm'][1],
        dtf['signal'][indxt,:] *scalet['hv'],
        'r*-'
        )


    ax4[1,1].set_xlabel('vert. bin [cm]')
    ax4[1,1].set_ylabel(label_hv)
    ax4[1,1].set_title('horz. bin %i/%i'%(indx, dxi['npix'][0]-1))
    ax4[1,1].grid('on')



    ###### ---- Used in paper ---- ######

    ymax = np.max((
        np.max(np.sum(dxi['signal'], axis=0))*scalex['v'],
        np.max(np.sum(dtf['signal'],axis=0))*scalet['v']
        ))

    fig3 = plt.figure(figsize = (16,10))
    gs3 = gridspec.GridSpec(2,3, width_ratios = [2,1, 1], hspace = 0.30, wspace = 0.3)
    ms = 8
    lw = 3
    pa = 20
    factor0 = 1e11
    factor1 = 1e10

    ##### PLots XICSRT detector image ######
    ax30 = fig3.add_subplot(gs3[1,0])

    im = ax30.imshow(
        dxi['signal'].T*scalex['hv'], # normalize [# ph] detected by [# ph] emitted
        extent = np.asarray(dxi['extent'])*100,
        interpolation='nearest',
        origin='lower',
        vmin=0,
        vmax=np.max(dxi['signal'].flatten())*scalex['hv'],
        aspect = 'equal'
        )

    ax30.set_title('XICSRT, #ph detected = %1.3e'%(
        np.sum(dxi['signal'].flatten())
        ), color = 'blue')
    ax30.set_xlabel('horiz. bin [cm]')
    ax30.set_ylabel('vert. bin [cm]')

    cb = fig3.colorbar(im, ax=ax30, orientation='vertical')
    cb.set_label(r'signal [%s]'%(label_hv))

    ax30.text(0.05, 0.90, '(b)', color = 'w', transform=ax30.transAxes)


    ##### PLots ToFu detector image ######
    ax31 = fig3.add_subplot(gs3[0,0])


    im1 = ax31.imshow(
        dtf['signal'].T*scalet['hv'],
        #extent = extent,
        extent = np.asarray(dtf['extent'])*100,
        interpolation='nearest',
        origin='lower',
        vmin=0,
        vmax=np.max(dtf['signal'].flatten())*scalet['hv'],
        aspect = 'equal'
        )
    ax31.set_title('ToFu, #ph detected = %1.3e'%(
        np.sum(dtf['signal'].flatten())
        ), color = 'red')
    ax31.set_xlabel('horiz. bin [cm]')
    ax31.set_ylabel('vert. bin [cm]')

    #cb = fig3.colorbar(im, ax=[ax30, ax31], orientation='vertical')
    cb = fig3.colorbar(im, ax=ax31, orientation='vertical')
    cb.set_label(r'signal [%s]'%(label_hv))

    ax31.text(0.05, 0.90, '(a)', color = 'w', transform=ax31.transAxes)

    ##### Plots image slice ######
    ax3 = fig3.add_subplot(gs3[0,1])

    ax3.plot(
        #dxi['cents_cm'][1],
        #dxi['signal'][xind,:],
        dxi['signal'][indx,:]*scalex['hv']*factor0,
        dxi['cents_cm'][1],
        'bD-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        #dtf['cents_cm'][1],
        #dtf['signal'][xind_tf,:]*scalet_0,
        dtf['signal'][indxt,:]*scalet['hv']*factor0,
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
    ax3.set_xlabel(r'signal [1e-11 %s]'%(label_hv))

    leg = ax3.legend(labelcolor='linecolor')
    leg.set_draggable('on')
    ax3.set_title('horiz. bin %0.0i/%0.0i'%(
            indx+1, dxi['npix'][0]
            ),
        pad = 0
        )

    #ax3.set_ylim(0,1.1*ymax)
    #ax3.set_xlim(0,1.1*ymax*factor0)

    ax3.text(0.15, 0.90, '(c)', color = 'k', transform=ax3.transAxes)


    ##### Plots integrated image ######
    ax3 = fig3.add_subplot(gs3[1,1])

    ax3.plot(
        #dxi['cents_cm'][1],
        #np.sum(dxi['signal'],axis=0),
        np.sum(dxi['signal'],axis=0)*scalex['v']*factor1,
        dxi['cents_cm'][1],
        'bD-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        #dtf['cents_cm'][1],
        #np.sum(dtf['signal'],axis=0)*scalet_1,
        np.sum(dtf['signal'],axis=0)*scalet['v']*factor1,
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
    ax3.set_xlabel('signal [1e-10 %s]'%(label_v))
    
    #ax3.set_ylim(0,1.1*ymax)
    #ax3.set_xlim(0,1.1*ymax*factor1)

    ax3.text(0.15, 0.90, '(d)', color = 'k', transform=ax3.transAxes)



    ##### Plots image slice ######
    ax3 = fig3.add_subplot(gs3[0,2])

    ax3.plot(
        dxi['cents_cm'][0],
        dxi['signal'][:,indy]*scalex['hv']*factor0,
        'bD-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        dtf['cents_cm'][0],
        dtf['signal'][:,indyt]*scalet['hv']*factor0,
        'ro-',
        label = 'ToFu',
        linewidth = lw,
        markersize = ms
        )

    ax3.grid('on')
    ax3.set_xlabel('horiz. bin [cm]')
    ax3.set_ylabel(r'signal [1e-11 %s]'%(label_hv))

    #leg = ax3.legend(labelcolor='linecolor')
    #leg.set_draggable('on')
    ax3.set_title('vert. bin %0.0i/%0.0i'%(
            indy+1, dxi['npix'][1]
            ),
        pad = 0
        )

    #ax3.set_ylim(0,1.1*ymax)
    #ax3.set_ylim(0,1.1*ymax*factor0)
    ax3.set_xlim(-0.25, -0.1)

    ax3.text(0.05, 0.90, '(e)', color = 'k', transform=ax3.transAxes)


    ##### Plots integrated image ######
    ax3 = fig3.add_subplot(gs3[1,2])

    ax3.plot(
        dxi['cents_cm'][0],
        np.sum(dxi['signal'],axis=1)*scalex['h']*factor1,
        'bD-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        dtf['cents_cm'][0],
        np.sum(dtf['signal'],axis=1)*scalet['h']*factor1,
        'ro-',
        label = 'ToFu',
        linewidth = lw,
        markersize = ms
        )

    ax3.set_title('int. over all vert. bin', pad = 0)
    ax3.grid('on')
    ax3.set_xlabel('horiz. bin [cm]')
    ax3.set_ylabel('signal [1e-10 %s]'%(label_h))
    
    #ax3.set_ylim(0,1.1*ymax*factor1)
    ax3.set_xlim(-0.25, -0.1)

    ax3.text(0.05, 0.90, '(f)', color = 'k', transform=ax3.transAxes)

    #fig2.suptitle('Detector binned (%0.0i, %0.0i), point = [%1.2f, %1.2f, %1.2f] m'%(
    #        dxi['npix'][0]-1, dxi['npix'][1]-1, 
    #        #lamb0, 
    #        dpt['ToFu']['point'][0], dpt['ToFu']['point'][1], dpt['ToFu']['point'][2]
    #        )
    #    )











