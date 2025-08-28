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

colors = ['b', 'r', 'g', 'm', 'c']
markers = ['D', 'o', 'D', 'o']


__all__ = [
    'plt_mono_vol',
    'plt_vos',
    ]

################################################
#
#               Utilities
#
################################################

# Plots detector image
def _plt_image(
    # Plot
    ax = None,
    tag = None,
    color = None,
    do_cbar = False,
    clabel = None,
    vmax = None,
    # Data
    scale = None,
    ddata = None,
    ):

    im1 = ax.imshow(
        ddata['signal'].T*scale['hv'],
        extent = np.asarray(ddata['extent'])*100,
        interpolation='nearest',
        origin='lower',
        vmin=0,
        vmax=vmax,
        #aspect = ddata['aspect']
        aspect = 'equal'
        )
    
    ax.set_title('%s, # ph detected = %1.5e'%(
        tag, 
        np.sum(ddata['signal'].flatten())
        ), color = color)
    ax.set_xlabel('horiz. bin [cm]')
    ax.set_ylabel('vert. bin [cm]')
    if do_cbar:
        cb = plt.colorbar(im1, ax=ax, orientation='vertical')
        cb.set_label('signal [%s]'%(clabel))

# Plots image slices
def _plt_slice(
    # Plotting
    ax = None,
    color = None,
    marker = None,
    tag = None,
    lw = None,
    ms = None,
    label = None,
    do_legend = False,
    do_title = False,
    do_zoom = False,
    dcm = 1, # [cm]
    # Controls
    do_horz_slice = False,
    do_horz_int = False,
    do_vert_slice = False,
    do_vert_int = False,
    ind = None,
    # Data
    ddata = None,
    scale = None,
    factor = None,
    ):

    # Look at a specific vertical slice
    if do_vert_slice:
        ax.plot(
            ddata['signal'][ind,:]*scale['hv']*factor,
            ddata['cents_cm'][1],
            '-',
            color = color,
            marker =  marker,
            label = tag,
            linewidth = lw,
            markersize = ms
            )

        ax.grid('on')
        ax.set_ylabel('vert. bin [cm]')
        ax.set_xlabel(r'signal [%1.e %s]'%(1/factor, label))

        if do_legend:
            leg = ax.legend(labelcolor='linecolor')
            leg.set_draggable('on')
        if do_title:
            ax.set_title('horiz. bin %0.0i/%0.0i'%(
                    ind+1, ddata['npix'][0]
                    ),
                pad = 0
                )

        if do_zoom:
            indz = np.argmax(ddata['signal'][ind,:])
            zmax = np.min(np.r_[
                np.max(ddata['cents_cm'][1])+np.mean(np.diff(ddata['cents_cm'][1]))/2,
                ddata['cents_cm'][1][indz] + dcm
                ])
            zmin = np.max(np.r_[
                np.min(ddata['cents_cm'][1])-np.mean(np.diff(ddata['cents_cm'][1]))/2,
                ddata['cents_cm'][1][indz] - dcm
                ])
            ax.set_ylim(zmin, zmax)

    # If integreated over all horizontal channels
    elif do_horz_int:
        ax.plot(
            np.sum(ddata['signal'],axis=0)*scale['v']*factor,
            ddata['cents_cm'][1],
            '-',
            color = color,
            marker = marker,
            label = tag,
            linewidth = lw,
            markersize = ms
            )

        ax.grid('on')
        ax.set_ylabel('vert. bin [cm]')
        ax.set_xlabel('signal [%1.e %s]'%(1/factor, label))
        
        if do_title:
            ax.set_title('int. over all horiz. bin', pad = 0)

        if do_zoom:
            indz = np.argmax(np.sum(ddata['signal'],axis=0))
            zmax = np.min(np.r_[
                np.max(ddata['cents_cm'][1])+np.mean(np.diff(ddata['cents_cm'][1])),
                ddata['cents_cm'][1][indz] + dcm
                ])
            zmin = np.max(np.r_[
                np.min(ddata['cents_cm'][1])-np.mean(np.diff(ddata['cents_cm'][1])),
                ddata['cents_cm'][1][indz] - dcm
                ])
            ax.set_ylim(zmin, zmax)

    # If looking at a slice through horizontal channel
    elif do_horz_slice:
        ax.plot(
            ddata['cents_cm'][0],
            ddata['signal'][:,ind]*scale['hv']*factor,
            '-',
            color = color,
            marker = marker,
            label = tag,
            linewidth = lw,
            markersize = ms
            )

        ax.grid('on')
        ax.set_xlabel('horiz. bin [cm]')
        ax.set_ylabel('signal [%1.e %s]'%(1/factor, label))

        if do_title:
            ax.set_title('vert. bin %0.0i/%0.0i'%(
                    ind+1, ddata['npix'][1]
                    ),
                pad = 0
                )

        if do_zoom:
            indz = np.argmax(ddata['signal'][:,ind])
            zmax = np.min(np.r_[
                np.max(ddata['cents_cm'][0])+np.mean(np.diff(ddata['cents_cm'][0])),
                ddata['cents_cm'][0][indz] + dcm
                ])
            zmin = np.max(np.r_[
                np.min(ddata['cents_cm'][0])-np.mean(np.diff(ddata['cents_cm'][0])),
                ddata['cents_cm'][0][indz] - dcm
                ])
            ax.set_xlim(zmin, zmax)

    #If integreated over all horizontal channels
    elif do_vert_int:
        ax.plot(
            ddata['cents_cm'][0],
            np.sum(ddata['signal'],axis=1)*scale['h']*factor,
            '-',
            color = color,
            marker = marker,
            label = tag,
            linewidth = lw,
            markersize = ms
            )

        ax.grid('on')
        ax.set_xlabel('horiz. bin [cm]')
        ax.set_ylabel('signal [%1.e %s]'%(1/factor, label))

        if do_title:
            ax.set_title('int. over all vert. bin', pad = 0)
        
        if do_zoom:
            indz = np.argmax(np.sum(ddata['signal'],axis=1))
            zmax = np.min(np.r_[
                np.max(ddata['cents_cm'][0])+np.mean(np.diff(ddata['cents_cm'][0])),
                ddata['cents_cm'][0][indz] + dcm
                ])
            zmin = np.max(np.r_[
                np.min(ddata['cents_cm'][0])-np.mean(np.diff(ddata['cents_cm'][0])),
                ddata['cents_cm'][0][indz] - dcm
                ])
            ax.set_xlim(zmin, zmax)

################################################
#
#               Main
#
################################################

# Plot VOS of diag
def plt_vos(
    ddatas = None,
    gfile = None,
    ):

    # Modules
    from transport_world.run_profiletools import eqtools3 as geq

    # Gets plasma equilbrium
    dedr, edr = geq._get_eq(
        gfile = gfile,
        machine = 'SPARC'
        )

    # Generates tokamak as a circle
    theta = np.linspace(0, 2*np.pi, 1001)

    rad_w0 = np.min(dedr['wall_R'])
    rad_w1 = np.max(dedr['wall_R'])

    X_w0 = rad_w0 *np.cos(theta)
    Y_w0 = rad_w0 *np.sin(theta)

    X_w1 = rad_w1 *np.cos(theta)
    Y_w1 = rad_w1 *np.sin(theta)

    Zmax = 9
    Zmin = -5
    R0 = 1.85*100
    aa = 0.57*100

    X1 = (R0-aa)*np.cos(theta)
    Y1 = (R0-aa)*np.sin(theta)

    X2 = (R0+aa)*np.cos(theta)
    Y2 = (R0+aa)*np.sin(theta)

    ##### Plotting
    fig1, ax1 = plt.subplots(2,2, figsize=(10,10))
    #plt.rcParams.update({'font.size': 14})
    #plt.subplots_adjust(wspace=0.4,hspace=0.4)
    lw = 2

    ### --- Plasma cross-section
    ax1[0,0].plot(
        dedr['wall_R']*100,
        dedr['wall_Z']*100,
        'k-',
        linewidth=lw
        )

    ax1[0,0].plot(
        dedr['RLCFS']*100,
        dedr['ZLCFS']*100,
        'm-',
        linewidth=lw
        )

    # Plots ToFu VOS
    cntr = 0
    for ii, ddata in enumerate(ddatas):
        if 'ToFu' not in ddata.keys():
            continue
        ax1[0,0].plot(
            ddata['ToFu']['pcross'][0]*100,
            ddata['ToFu']['pcross'][1]*100,
            '-',
            color = colors[cntr]
            )

        cntr += 1

    ax1[0,0].grid('on')
    ax1[0,0].set_xlabel('R [cm]')
    ax1[0,0].set_ylabel('Z [cm]')
    ax1[0,0].set_aspect('equal')

    ### --- Plots top view of machine
    ax1[0,1].plot(X_w0*100, Y_w0*100, 'k-')
    ax1[0,1].plot(X_w1*100, Y_w1*100, 'k-')

    ax1[0,1].plot(X1, Y1, 'm-')
    ax1[0,1].plot(X2, Y2, 'm-')

    # Plots ToFu VOS
    cntr = 0
    for ii, ddata in enumerate(ddatas):
        if 'ToFu' not in ddata.keys():
            continue
        ax1[0,1].plot(
            ddata['ToFu']['phor'][0]*100,
            ddata['ToFu']['phor'][1]*100,
            '-',
            color = colors[cntr]
            )

        cntr += 1

    ax1[0,1].set_xlim(110,250)
    ax1[0,1].set_ylim(0,35)

    ax1[0,1].grid('on')
    ax1[0,1].set_xlabel('X [cm]')
    ax1[0,1].set_ylabel('Y [cm]')
    ax1[0,1].set_aspect('equal')



# Plots point source results
def plt_mono_vol(
    ddatas = None,
    ):

    ###########
    # --- Prep work
    ##########

    # Init
    scale = {}

    # Loop over cases
    for ii, ddata in enumerate(ddatas):
        # Init
        scale[ii] = {}

        # Loop over codes
        for kk in ['ToFu', 'XICSRT']:
            # Init
            scale[ii][kk] = {}

            # Normalizing by bin length
            scale[ii][kk]['h'] = (
                1
                /np.mean(np.diff(ddata[kk]['cents_cm'][0]))
                ) # [1/cm], horizontal bin width
            scale[ii][kk]['v'] = (
                1
                /np.mean(np.diff(ddata[kk]['cents_cm'][1]))
                ) # [1/cm], vertical bin width
            scale[ii][kk]['hv'] = scale[ii][kk]['h']*scale[ii][kk]['v'] # [1/cm^2], bin area

            # Maxima
            if ii == 0 and kk == 'ToFu':
                vmax_hv = np.max(ddata[kk]['signal'].flatten())*scale[ii][kk]['hv']

            # Indexing
            if ii == 0 and kk == 'ToFu':
                indyt = int(ddata[kk]['npix'][1]/2-1)
                indxt = np.argmax(np.sum(ddata[kk]['signal'], axis=1))
            elif ii == 0 and kk == 'XICSRT':
                indy = np.argmin(abs(
                    ddata['ToFu']['cents_cm'][1][indyt] - ddata[kk]['cents_cm'][1]
                    ))
                indx = np.argmin(abs(
                    ddata['ToFu']['cents_cm'][0][indxt] - ddata[kk]['cents_cm'][0]
                    ))

    # Labels
    label_hv = r'#$ph/s/cm^2$'
    label_h = label_v = r'#$ph/s/cm$'

    ###########
    # --- Detector images
    ##########

    # Plotting setting
    ms = 8
    lw = 3
    pa = 20

    # Rescaling
    factor_slice = 1e7
    factor_int = 1e8

    # Zooming
    dcm_h = 0.3
    dcm_v = 10

    # Plots photon flux on detector from XICSRT
    fig1, ax1 = plt.subplots(len(ddatas),2)

    cntr = 0
    for ii, ddata in enumerate(ddatas):
        for jj, kk in enumerate(['XICSRT', 'ToFu']):
            if kk not in ddata.keys():
                continue

            if len(ddatas) > 1:
                ax = ax1[ii,jj]
            else:
                ax = ax1[jj]

            _plt_image(
                ax = ax,
                tag = kk,
                color = colors[cntr],
                do_cbar = jj == len(ddata.keys())-1,
                clabel = label_hv,
                vmax = vmax_hv,
                scale = scale[ii][kk],
                ddata = ddata[kk],
                )

            cntr += 1

    fig1.show()

    for ddata in ddatas:
        print('Total Flux Error:')
        print('%0.2f %%'%(
            (
                1
                -np.sum(ddata['XICSRT']['signal'].flatten())
                /np.sum(ddata['ToFu']['signal'].flatten())
                )*100
            )) #

    ###########
    # --- Image Slices
    ##########

    fig4, ax4 = plt.subplots(2,2)

    cntr = 0
    for ii, ddata in enumerate(ddatas):
        for jj, kk in enumerate(['XICSRT', 'ToFu']):
            if kk not in ddata.keys():
                continue

            #### Plots slice integrated over vertical channels
            _plt_slice(
                # Plotting
                ax = ax4[0,0],
                color = colors[cntr],
                marker = markers[cntr],
                tag = kk,
                lw = lw,
                ms = ms,
                label = label_h,
                do_title = ii==0 and kk == 'XICSRT',
                do_zoom = ii==0 and kk == 'XICSRT',
                dcm = dcm_h,
                # Controls
                do_vert_int = True,
                # Data
                ddata = ddata[kk],
                scale = scale[ii][kk],
                factor = factor_int,
                )

            #### Plots slice integrated over horizontal channels 
            _plt_slice(
                # Plotting
                ax = ax4[0,1],
                color = colors[cntr],
                marker = markers[cntr],
                tag = kk,
                lw = lw,
                ms = ms,
                label = label_v,
                do_title = ii==0 and kk == 'XICSRT',
                do_zoom = ii==0 and kk == 'XICSRT',
                dcm = dcm_v,
                # Controls
                do_horz_int = True,
                # Data
                ddata = ddata[kk],
                scale = scale[ii][kk],
                factor = factor_int,
                )

            #### Plots horizontal slice
            _plt_slice(
                # Plotting
                ax = ax4[1,0],
                color = colors[cntr],
                marker = markers[cntr],
                tag = kk,
                lw = lw,
                ms = ms,
                label = label_hv,
                do_title = ii==0 and kk == 'XICSRT',
                do_zoom = ii==0 and kk == 'XICSRT',
                dcm = dcm_h,
                # Controls
                do_horz_slice = True,
                ind = indy if kk == 'XICSRT' else indyt,
                # Data
                ddata = ddata[kk],
                scale = scale[ii][kk],
                factor = factor_slice,
                )

            ### Plots vertical slice
            _plt_slice(
                # Plotting
                ax = ax4[1,1],
                color = colors[cntr],
                marker = markers[cntr],
                tag = kk,
                lw = lw,
                ms = ms,
                label = label_hv,
                do_legend = ii==len(ddatas)-1 and kk == 'ToFu',
                do_title = ii==0 and kk == 'XICSRT',
                do_zoom = ii==0 and kk == 'XICSRT',
                dcm = dcm_v,
                # Controls
                do_vert_slice = True,
                ind = indx if kk == 'XICSRT' else indxt,
                # Data
                ddata = ddata[kk],
                scale = scale[ii][kk],
                factor = factor_slice,
                )

            cntr += 1


    ###### ---- Used in paper ---- ######
    fig3 = plt.figure(figsize = (16,10))
    gs3 = gridspec.GridSpec(2,3, width_ratios = [2,1, 1], hspace = 0.30, wspace = 0.3)

    ddata = ddatas[0]
    ii = 0

    ##### PLots XICSRT detector image ######
    ax30 = fig3.add_subplot(gs3[1,0])

    kk = 'XICSRT'
    _plt_image(
        ax = ax30,
        tag = kk,
        color = 'blue',
        do_cbar = True,
        clabel = label_hv,
        vmax = vmax_hv,
        scale = scale[ii][kk],
        ddata = ddata[kk],
        )

    ax30.text(0.05, 0.90, '(b)', color = 'w', transform=ax30.transAxes)


    ##### PLots ToFu detector image ######
    ax31 = fig3.add_subplot(gs3[0,0])

    kk = 'ToFu'
    _plt_image(
        ax = ax31,
        tag = kk,
        color = 'red',
        do_cbar = True,
        clabel = label_hv,
        vmax = vmax_hv,
        scale = scale[ii][kk],
        ddata = ddata[kk],
        )

    ax31.text(0.05, 0.90, '(a)', color = 'w', transform=ax31.transAxes)

    ##### Plots image slice ######
    ax3 = fig3.add_subplot(gs3[0,1])

    cntr = 0
    for jj, kk in enumerate(['XICSRT', 'ToFu']):
        if kk not in ddata.keys():
            continue

        _plt_slice(
            # Plotting
            ax = ax3,
            color = colors[cntr],
            marker = markers[cntr],
            tag = kk,
            lw = lw,
            ms = ms,
            label = label_hv,
            do_legend = kk == 'ToFu',
            do_title = kk == 'XICSRT',
            do_zoom = kk == 'XICSRT',
            dcm = dcm_v,
            # Controls
            do_vert_slice = True,
            ind = indx if kk == 'XICSRT' else indxt,
            # Data
            ddata = ddata[kk],
            scale = scale[ii][kk],
            factor = factor_slice,
            )

        cntr += 1

    ax3.text(0.15, 0.90, '(c)', color = 'k', transform=ax3.transAxes)

    ##### Plots integrated image ######
    ax3 = fig3.add_subplot(gs3[1,1])

    cntr = 0
    for jj, kk in enumerate(['XICSRT', 'ToFu']):
        if kk not in ddata.keys():
            continue

        _plt_slice(
            # Plotting
            ax = ax3,
            color = colors[cntr],
            marker = markers[cntr],
            tag = kk,
            lw = lw,
            ms = ms,
            label = label_hv,
            do_title = kk == 'XICSRT',
            do_zoom = kk == 'XICSRT',
            dcm = dcm_v,
            # Controls
            do_horz_int = True,
            # Data
            ddata = ddata[kk],
            scale = scale[ii][kk],
            factor = factor_int,
            )

        cntr += 1

    ax3.text(0.15, 0.90, '(d)', color = 'k', transform=ax3.transAxes)

    ##### Plots image slice ######
    ax3 = fig3.add_subplot(gs3[0,2])

    cntr = 0
    for jj, kk in enumerate(['XICSRT', 'ToFu']):
        if kk not in ddata.keys():
            continue

        _plt_slice(
            # Plotting
            ax = ax3,
            color = colors[cntr],
            marker = markers[cntr],
            tag = kk,
            lw = lw,
            ms = ms,
            label = label_hv,
            do_title = kk == 'XICSRT',
            do_zoom = kk == 'XICSRT',
            dcm = dcm_h,
            # Controls
            do_horz_slice = True,
            ind = indy if kk == 'XICSRT' else indyt,
            # Data
            ddata = ddata[kk],
            scale = scale[ii][kk],
            factor = factor_slice,
            )

        cntr += 1

    ax3.text(0.05, 0.90, '(e)', color = 'k', transform=ax3.transAxes)

    ##### Plots integrated image ######
    ax3 = fig3.add_subplot(gs3[1,2])

    cntr = 0
    for jj, kk in enumerate(['XICSRT', 'ToFu']):
        if kk not in ddata.keys():
            continue

        _plt_slice(
            # Plotting
            ax = ax3,
            color = colors[cntr],
            marker = markers[cntr],
            tag = kk,
            lw = lw,
            ms = ms,
            label = label_hv,
            do_title = kk == 'XICSRT',
            do_zoom = kk == 'XICSRT',
            dcm = dcm_h,
            # Controls
            do_vert_int = True,
            # Data
            ddata = ddata[kk],
            scale = scale[ii][kk],
            factor = factor_int,
            )

        cntr += 1

    ax3.text(0.05, 0.90, '(f)', color = 'k', transform=ax3.transAxes)

    #fig2.suptitle('Detector binned (%0.0i, %0.0i), point = [%1.2f, %1.2f, %1.2f] m'%(
    #        dxi['npix'][0]-1, dxi['npix'][1]-1, 
    #        #lamb0, 
    #        dpt['ToFu']['point'][0], dpt['ToFu']['point'][1], dpt['ToFu']['point'][2]
    #        )
    #    )











