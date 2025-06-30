'''

Functions to plot comparison between ToFu and XICSRT for
an emissivity map

cjperks
Dec 16, 2024

'''

# Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Rectangle
import tofu as tf

plt.rcParams.update({'font.size': 16})
formatter = ScalarFormatter()
formatter.set_powerlimits((-2, 2))  # Use exponential notation if values are outside this range

import cmp_tofu_xicsrt.utils as utils

__all__ = [
    'plt_rad_emis',
    'plt_rad_emis_standalone',
    ]

################################################
#
#               Main
#
################################################


# Plots point source results
def plt_rad_emis(
    fdata = None,
    lamb0 = [0.945],
    coll_tf = None,
    key_diags = None,
    key_cams = None,
    norm_by_bin = True,
    dt = 1, # [s], integration time
    ):

    ###########
    # --- Detector images
    ##########

    # Init
    dxi = {}
    dtf = {}
    scalex = {}
    scalet = {}
    vmaxx = 0
    vmaxt = 0
    cntsx = 0
    cntst = 0
    for ii, ff in enumerate(fdata):
        ddata = np.load(fdata[ii], allow_pickle=True)['arr_0'][()]

        dxi[ii] = ddata['XICSRT']
        dtf[ii] = ddata['ToFu']

        scalex[ii] = {}
        scalet[ii] = {}

        # If normalizing by bin length
        if norm_by_bin:
            scalex[ii]['h'] = (
                1
                /np.mean(np.diff(dxi[ii]['cents_cm'][0]))
                ) # [1/cm], horizontal bin width
            scalet[ii]['h'] = (
                1
                /np.mean(np.diff(dtf[ii]['cents_cm'][0]))
                ) # [1/cm], horizontal bin width
            scalex[ii]['v'] = (
                1
                /np.mean(np.diff(dxi[ii]['cents_cm'][1]))
                ) # [1/cm], vertical bin width
            scalet[ii]['v'] = (
                1
                /np.mean(np.diff(dtf[ii]['cents_cm'][1]))
                ) # [1/cm], vertical bin width
            scalex[ii]['hv'] = scalex[ii]['h']*scalex[ii]['v'] # [1/cm^2], bin area
            scalet[ii]['hv'] = scalet[ii]['h']*scalet[ii]['v'] # [1/cm^2], bin area

            # Time-integration
            scalex[ii]['v'] *= dt
            scalex[ii]['h'] *= dt
            scalex[ii]['hv'] *= dt
            scalet[ii]['v'] *= dt
            scalet[ii]['h'] *= dt
            scalet[ii]['hv'] *= dt

            label_hv = r'#$ph/cm^2$'
            label_h = label_v = r'#$ph/cm$'
        else:
            # Time-integration
            scalex[ii]['h'] = scalex[ii]['v'] = scalex[ii]['hv'] = dt
            scalet[ii]['h'] = scalet[ii]['v'] = scalet[ii]['hv'] = dt

            label_hv = r'#$ph/bin^2$'
            label_h = label_v = r'#$ph/bin$'

        vmaxt = np.max(np.r_[vmaxt, np.max(dtf[ii]['signal'].flatten())*scalet[ii]['hv']])
        cntst += np.sum(dtf[ii]['signal'].flatten()*dt)

        vmaxx = np.max(np.r_[vmaxx, np.max(dxi[ii]['signal'].flatten())*scalex[ii]['hv']])
        cntsx += np.sum(dxi[ii]['signal'].flatten()*dt)


    # Plots photon flux on detector from XICSRT
    fig, ax = plt.subplots(1,2)
    
    tmp = np.zeros_like(dxi[ii]['signal'])
    for ii in dxi.keys():
        tmp += dxi[ii]['signal']*scalex[ii]['hv']
    tmp[tmp == 0] = np.nan
    ii = 0
    im = ax[1].imshow(
        tmp.T, # normalize [# ph] detected by [# ph] emitted
        extent = np.asarray(dxi[ii]['extent'])*100,
        interpolation='nearest',
        origin='lower',
        vmin=0,
        vmax=vmaxt,
        aspect = 1,
        )
    cb = plt.colorbar(im, ax=ax[1], orientation='horizontal')
    ax[1].set_title('XICSRT, # ph detected = %1.5e'%(cntsx), color = 'blue')
    ax[1].set_xlabel('horz. bin [cm]')
    cb.set_label(label_hv)
    cb.ax.yaxis.set_major_formatter(formatter)
    cb.formatter.set_scientific(True)
    cb.formatter.set_powerlimits((-2, 2))  # Always use exponential notation
    cb.update_ticks()  # Update ticks to apply the formatter

    x1, y1 = -3.855, -0.285/2 # [cm], Bottom-left corner
    x2, y2 = 3.855, 0.285/2 # [cm], Top-right corner
    width = x2 - x1
    height = y2 - y1
    rect1 = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='red')
    ax[1].add_patch(rect1)

    for ii in dtf.keys():
        tmp = np.zeros_like(dtf[ii]['signal'])
        tmp += dtf[ii]['signal']*scalet[ii]['hv']
        tmp[tmp == 0] = np.nan

        im1 = ax[0].imshow(
            #dtf[ii]['signal'].T*scalet[ii]['hv'],
            tmp.T,
            #extent = extent,
            extent = np.asarray(dtf[ii]['extent'])*100,
            interpolation='nearest',
            origin='lower',
            vmin=0,
            vmax=vmaxt,
            aspect = dtf[ii]['aspect']
            )
    cb = plt.colorbar(im1, ax=ax[0], orientation='horizontal')
    ax[0].set_title('ToFu, # ph detected = %1.5e'%(cntst), color = 'red')
    ax[0].set_xlabel('horz. bin [cm]')
    ax[0].set_ylabel('vert. bin [cm]')
    cb.set_label(label_hv)
    cb.formatter.set_scientific(True)
    cb.formatter.set_powerlimits((-2, 2))  # Always use exponential notation
    cb.update_ticks()  # Update ticks to apply the formatter

    ax[0].set_xlim(-3.855,3.855)
    ax[0].set_ylim(-3.9825,3.9825)
    rect2 = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='red')
    ax[0].add_patch(rect2)
    ax[0].set_aspect('equal')

    fig.show()

    '''
    print('Total Flux Error:')
    print('%0.2f %%'%(
        (1-np.sum(dxi['signal'].flatten())/np.sum(dtf['signal'].flatten()))*100
        )) #
    '''

    ###########
    # --- Image Slices
    ##########
    
    for ii, kk in enumerate(key_diags):

        # Overplotting data
        if kk == 'XRSHRKr':
            lines = {
                'w': 0.9454,
                '(x,s)': 0.9471,
                't': 0.94804,
                '(q,k,a)': 0.94961,
                '(y,j)': 0.9518,
                '(r,B)':0.95288,
                'z': 0.9552,
                }
            calib = {
                'Sb Ka1 (2nd)': 0.940736,
                'Sb Ka2 (2nd)': 0.949681
                }
        elif kk == 'XRSHRXe':
            lines = {
                '3D': 2.7204,
                '3F': 2.7290,
                'Na': 2.7364,
                'Mg': 2.7472
                }
            calib = {}
        elif kk == 'XRSLR':
            lines = {
                'W: 3G': 1.49282,
                }
            calib = {
                'Cu Ka1': 1.540607,
                'Cu Ka2': 1.544436,
            }

        # Loads Collection object
        coll = tf.data.load(coll_tf[ii])

        # Gets wavelength basis
        lamb_tf, _ = coll.get_diagnostic_lamb(
            kk,
            key_cam=key_cams[ii],
            lamb='lamb',
            ) # dim(nx, ny)

        # Indexing
        indyt = int(dtf[ii]['npix'][1]/2-1)
        indxt = np.nanargmin(abs(
            lamb_tf[:,indyt] - lamb0[ii]*1e-10
            ))

        indx = np.argmin(abs(
            dxi[ii]['cents_cm'][0] - dtf[ii]['cents_cm'][0][indxt]
            ))
        indy = np.argmin(abs(
            dxi[ii]['cents_cm'][1] - dtf[ii]['cents_cm'][1][indyt]
            ))

        # Plotting
        fig4, ax4 = plt.subplots(2,2)

        ax4[0,0].plot(
            dxi[ii]['cents_cm'][0],
            np.sum(dxi[ii]['signal'], axis = 1) *scalex[ii]['h'],
            'b*-',
            label = 'XICSRT'
            )

        tmp = np.sum(dtf[ii]['signal'], axis = 1) *scalet[ii]['h']
        ax4[0,0].plot(
            dtf[ii]['cents_cm'][0],
            tmp,
            'r*-',
            label = 'ToFu'
            )
        if not norm_by_bin:
            ax4[0,0].fill_between(
                dtf[ii]['cents_cm'][0],
                tmp - np.sqrt(tmp),
                tmp + np.sqrt(tmp),
                color = 'r',
                alpha = 0.6
                )
            ax4[0,0].fill_between(
                dtf[ii]['cents_cm'][0],
                tmp - 2*np.sqrt(tmp),
                tmp + 2*np.sqrt(tmp),
                color = 'r',
                alpha = 0.3
                )

        for ll in lines.keys():
            indlt = np.nanargmin(abs(
                lamb_tf[:,indyt] - lines[ll]*1e-10
                ))

            ax4[0,0].text(
                dtf[ii]['cents_cm'][0][indlt],
                (np.sum(dtf[ii]['signal'], axis = 1) *scalet[ii]['h'])[indlt],
                ll,
                color = 'k'
                )
        for ll in calib.keys():
            indlt = np.nanargmin(abs(
                lamb_tf[:,indyt] - calib[ll]*1e-10
                ))
            ax4[0,0].axvspan(
                dtf[ii]['cents_cm'][0][indlt]-0.05,
                dtf[ii]['cents_cm'][0][indlt]+0.05,
                color = 'm',
                alpha = 0.3
                )
            ax4[0,0].text(
                dtf[ii]['cents_cm'][0][indlt],
                np.nanmax(np.sum(dtf[ii]['signal'], axis = 1) *scalet[ii]['h']),
                ll,
                color = 'k'
                )

        ax4[0,0].set_xlabel('horiz. bin [cm]')
        ax4[0,0].set_ylabel(label_h)
        ax4[0,0].yaxis.set_major_formatter(formatter)
        ax4[0,0].set_title('int. over all vert. bins')
        ax4[0,0].grid('on')
        #ax4[0,0].legend(labelcolor='linecolor')
        ax4[0,0].set_xlim(0.95*np.min(dtf[ii]['cents_cm'][0]), 1.05*np.max(dtf[ii]['cents_cm'][0]))

        ax4[1,0].plot(
            dxi[ii]['cents_cm'][0],
            dxi[ii]['signal'][:,indy] *scalex[ii]['hv'],
            'b*-',
            label = 'XICSRT'
            )

        tmp = dtf[ii]['signal'][:,indyt] *scalet[ii]['hv']
        ax4[1,0].plot(
            dtf[ii]['cents_cm'][0],
            tmp,
            'r*-',
            label = 'ToFu'
            )
        if not norm_by_bin:
            ax4[1,0].fill_between(
                dtf[ii]['cents_cm'][0],
                tmp - np.sqrt(tmp),
                tmp + np.sqrt(tmp),
                color = 'r',
                alpha = 0.6
                )
            ax4[1,0].fill_between(
                dtf[ii]['cents_cm'][0],
                tmp - 2*np.sqrt(tmp),
                tmp + 2*np.sqrt(tmp),
                color = 'r',
                alpha = 0.3
                )

        ax4[1,0].set_xlabel('horiz. bin [cm]')
        ax4[1,0].set_ylabel(label_hv)
        ax4[1,0].yaxis.set_major_formatter(formatter)
        #ax4[1,0].set_title('vert. bin %i/%i'%(indy, dxi['npix'][1]-1))
        #ax4[1,0].set_title('vert. bin %i/%i'%(indyt, dtf['npix'][1]-1))
        ax4[1,0].set_title('vert. bin @ %0.2f cm'%(dtf[ii]['cents_cm'][1][indyt]))
        ax4[1,0].grid('on')
        ax4[1,0].set_xlim(0.95*np.min(dtf[ii]['cents_cm'][0]), 1.05*np.max(dtf[ii]['cents_cm'][0]))
        ax4[1,0].legend(labelcolor='linecolor')

        ax4[0,1].plot(
            np.sum(dxi[ii]['signal'], axis = 0) *scalex[ii]['v'],
            dxi[ii]['cents_cm'][1],
            'b*-'
            )

        ax4[0,1].plot(
            np.sum(dtf[ii]['signal'], axis = 0) *scalet[ii]['v'],
            dtf[ii]['cents_cm'][1],
            'r*-'
            )

        if np.min(dtf[ii]['cents_cm'][1]) < 0:
            ff1 = 1.05
        else:
            ff1 = 0.95
        if np.max(dtf[ii]['cents_cm'][1]) < 0:
            ff2 = 0.95
        else:
            ff2 = 1.05

        ax4[0,1].set_ylabel('vert. bin [cm]')
        ax4[0,1].set_xlabel(label_v)
        ax4[0,1].xaxis.set_major_formatter(formatter)
        ax4[0,1].set_title('int. over all horz. bins')
        ax4[0,1].grid('on')
        ax4[0,1].set_ylim(ff1*np.min(dtf[ii]['cents_cm'][1]), ff2*np.max(dtf[ii]['cents_cm'][1]))

        ax4[1,1].plot(
            dxi[ii]['signal'][indx,:] *scalex[ii]['hv'],
            dxi[ii]['cents_cm'][1],
            'b*-'
            )

        ax4[1,1].plot(
            dtf[ii]['signal'][indxt,:] *scalet[ii]['hv'],
            dtf[ii]['cents_cm'][1],
            'r*-'
            )


        ax4[1,1].set_ylabel('vert. bin [cm]')
        ax4[1,1].set_xlabel(label_hv)
        ax4[1,1].xaxis.set_major_formatter(formatter)
        #ax4[1,1].set_title('horz. bin %i/%i'%(indx, dxi['npix'][0]-1))
        #ax4[1,1].set_title('horz. bin %i/%i'%(indxt, dtf['npix'][0]-1))
        ax4[1,1].set_title('horz. bin @ %0.2f cm'%(dtf[ii]['cents_cm'][0][indxt]))
        ax4[1,1].grid('on')
        ax4[1,1].set_ylim(ff1*np.min(dtf[ii]['cents_cm'][1]), ff2*np.max(dtf[ii]['cents_cm'][1]))


    '''
    ###### ---- Used in paper ---- ######

    # Gets wavelength map from XICSRT
    dxi = utils._get_dispersion_xicsrt(dxi=dxi, lamb0=lamb0)

    ymax = np.max((
        np.max(np.sum(dxi['signal'], axis=0)),
        np.max(np.sum(dtf['signal'],axis=0))*scalet_1
        ))

    fig3 = plt.figure(figsize = (16,10))
    gs3 = gridspec.GridSpec(2,3, width_ratios = [2,1, 1], hspace = 0.30, wspace = 0.30)
    ms = 8
    lw = 3
    pa = 20
    factor0 = 1e11
    factor1 = 1e10

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
        dxi['signal'][indx,:]*factor0,
        dxi['cents_cm'][1],
        'bD-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        #dtf['cents_cm'][1],
        #dtf['signal'][xind_tf,:]*scalet_0,
        dtf['signal'][indxt,:]*scalet_0*factor0,
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
    ax3.set_xlabel(r'signal [1e-11 #ph/bin$^2$]')

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
        np.sum(dxi['signal'],axis=0)*factor1,
        dxi['cents_cm'][1],
        'bD-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        #dtf['cents_cm'][1],
        #np.sum(dtf['signal'],axis=0)*scalet_1,
        np.sum(dtf['signal'],axis=0)*scalet_1*factor1,
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
    ax3.set_xlabel('signal [1e-10 #ph/bin]')
    
    #ax3.set_ylim(0,1.1*ymax)
    #ax3.set_xlim(0,1.1*ymax*factor1)

    ax3.text(0.15, 0.90, '(d)', color = 'k', transform=ax3.transAxes)



    ##### Plots image slice ######
    ax3 = fig3.add_subplot(gs3[0,2])

    ax3.plot(
        dxi['cents_cm'][0],
        dxi['signal'][:,indy]*factor0,
        'bD-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        dtf['cents_cm'][0],
        dtf['signal'][:,indyt]*scalet_0*factor0,
        'ro-',
        label = 'ToFu',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        dxi['cents_cm'][0],
        (
            dxi['lambda_pix']['gaussian'][:,indy]
            /np.nanmax(dxi['lambda_pix']['gaussian'][:,indy])
            *np.nanmax(dxi['signal'][:,indy]*factor0)
            ),
        'k-',
        label = 'Init',
        linewidth = lw-1,
        markersize = ms
        )

    ax3.grid('on')
    ax3.set_xlabel('horiz. bin [cm]')
    ax3.set_ylabel(r'signal [1e-11 #ph/bin$^2$]')

    #leg = ax3.legend(labelcolor='linecolor')
    #leg.set_draggable('on')
    ax3.set_title('vert. bin %0.0i/%0.0i'%(
            indy+1, dxi['npix'][1]
            ),
        pad = 0
        )

    #ax3.set_ylim(0,1.1*ymax)
    #ax3.set_ylim(0,1.1*ymax*factor0)
    #ax3.set_xlim(-0.25, -0.1) # sph_me
    #ax3.set_xlim(0, 0.5) # cyl_me

    ax3.text(0.05, 0.90, '(e)', color = 'k', transform=ax3.transAxes)


    ##### Plots integrated image ######
    ax3 = fig3.add_subplot(gs3[1,2])

    ax3.plot(
        dxi['cents_cm'][0],
        np.sum(dxi['signal'],axis=1)*factor1,
        'bD-',
        label = 'XICSRT',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        dtf['cents_cm'][0],
        np.sum(dtf['signal'],axis=1)*scalet_00*factor1,
        'ro-',
        label = 'ToFu',
        linewidth = lw,
        markersize = ms
        )
    ax3.plot(
        dxi['cents_cm'][0],
        (
            dxi['lambda_pix']['gaussian'][:,indy]
            /np.nanmax(dxi['lambda_pix']['gaussian'][:,indy])
            *np.nanmax(np.sum(dxi['signal'],axis=1)*factor1)
            ),
        'k-',
        label = 'Init',
        linewidth = lw-1,
        markersize = ms
        )

    ax3.set_title('int. over all vert. bin', pad = 0)
    ax3.grid('on')
    ax3.set_xlabel('horiz. bin [cm]')
    ax3.set_ylabel('signal [1e-10 #ph/bin]')
    
    #ax3.set_ylim(0,1.1*ymax*factor1)
    #ax3.set_xlim(-0.25, -0.1) # sph_me
    #ax3.set_xlim(0, 0.5) # cyl_me

    ax3.text(0.05, 0.90, '(f)', color = 'k', transform=ax3.transAxes)

    #fig2.suptitle('Detector binned (%0.0i, %0.0i), point = [%1.2f, %1.2f, %1.2f] m'%(
    #        dxi['npix'][0]-1, dxi['npix'][1]-1, 
    #        #lamb0, 
    #        dpt['ToFu']['point'][0], dpt['ToFu']['point'][1], dpt['ToFu']['point'][2]
    #        )
    #    )
    '''



################################################
#
#               Extra
#
################################################

# Plots point source results
def plt_rad_emis_standalone(
    fdata = None,
    lamb0 = [0.945],
    key_data = 'XICSRT',
    color_data = 'blue',
    coll_tf = None,
    key_diags = None,
    key_cams = None,
    norm_by_bin = True,
    dt = 1, # [s], integration time
    # Other data
    other_data = '',
    vmaxx_user = None,
    ):

    ###########
    # --- Detector images
    ##########

    # Init
    dxi = {}
    scalex = {}
    vmaxx = 0
    cntsx = 0
    for ii, ff in enumerate(fdata):
        ddata = np.load(fdata[ii], allow_pickle=True)['arr_0'][()]
        kk = key_diags[ii]
        dxi[kk] = ddata[key_data]

        scalex[kk] = {}

        # If normalizing by bin length
        if norm_by_bin:
            scalex[kk]['h'] = (
                1
                /np.mean(np.diff(dxi[kk]['cents_cm'][0]))
                ) # [1/cm], horizontal bin width
            scalex[kk]['v'] = (
                1
                /np.mean(np.diff(dxi[kk]['cents_cm'][1]))
                ) # [1/cm], vertical bin width
            scalex[kk]['hv'] = scalex[kk]['h']*scalex[kk]['v'] # [1/cm^2], bin area

            # Time-integration
            scalex[kk]['v'] *= dt
            scalex[kk]['h'] *= dt
            scalex[kk]['hv'] *= dt

            label_hv = r'#$ph/cm^2$'
            label_h = label_v = r'#$ph/cm$'
        else:
            # Time-integration
            scalex[kk]['h'] = scalex[kk]['v'] = scalex[kk]['hv'] = dt

            label_hv = r'#$ph/bin^2$'
            label_h = label_v = r'#$ph/bin$'

        vmaxx = np.max(np.r_[vmaxx, np.max(dxi[kk]['signal'].flatten())*scalex[kk]['hv']])
        cntsx += np.sum(dxi[kk]['signal'].flatten()*dt)

    if vmaxx_user is not None:
        vmaxx = vmaxx_user

    # Plots photon flux on detector from XICSRT
    fig, ax = plt.subplots()
    
    if key_data == 'XICSRT':
        tmp = np.zeros_like(dxi[ii]['signal'])
        for ii in dxi.keys():
            tmp += dxi[ii]['signal']*scalex[ii]['hv']
        tmp[tmp == 0] = np.nan
        ii = 0
        im = ax.imshow(
            tmp.T, # normalize [# ph] detected by [# ph] emitted
            extent = np.asarray(dxi[ii]['extent'])*100,
            interpolation='nearest',
            origin='lower',
            vmin=0,
            vmax=vmaxx,
            aspect = 1,
            )

    elif key_data == 'ToFu':
        for ii in dxi.keys():
            tmp = np.zeros_like(dxi[ii]['signal'])
            tmp += dxi[ii]['signal']*scalex[ii]['hv']
            tmp[tmp == 0] = np.nan

            im = ax.imshow(
                #dtf[ii]['signal'].T*scalet[ii]['hv'],
                tmp.T,
                #extent = extent,
                extent = np.asarray(dxi[ii]['extent'])*100,
                interpolation='nearest',
                origin='lower',
                vmin=0,
                vmax=vmaxx,
                aspect = 1,
                )

    cb = plt.colorbar(im, ax=ax, orientation='horizontal')
    ax.set_title(key_data+', # ph detected = %1.5e'%(cntsx), color = color_data)
    ax.set_xlabel('horz. bin [cm]')
    ax.set_ylabel('vert. bin [cm]')
    cb.set_label(label_hv)
    cb.ax.yaxis.set_major_formatter(formatter)
    cb.formatter.set_scientific(True)
    cb.formatter.set_powerlimits((-2, 2))  # Always use exponential notation
    cb.update_ticks()  # Update ticks to apply the formatter

    ax.set_xlim(-3.855,3.855)
    ax.set_ylim(-3.9825,3.9825)

    x1, y1 = -3.855, -0.285/2 # [cm], Bottom-left corner
    x2, y2 = 3.855, 0.285/2 # [cm], Top-right corner
    width = x2 - x1
    height = y2 - y1
    rect1 = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='red')
    ax.add_patch(rect1)

    fig.show()

    '''
    print('Total Flux Error:')
    print('%0.2f %%'%(
        (1-np.sum(dxi['signal'].flatten())/np.sum(dtf['signal'].flatten()))*100
        )) #
    '''

    ###########
    # --- Image Slices
    ##########
    
    for ii, kk in enumerate(key_diags):

        # Overplotting data
        if 'XRSHRKr' in kk:
            lines = {
                'w': 0.9454,
                '(x,s)': 0.9471,
                't': 0.94804,
                '(q,k,a)': 0.94961,
                '(y,j)': 0.9518,
                '(r,B)':0.95288,
                'z': 0.9552,
                }
            calib = {
                'Sb Ka1 (2nd)': 0.940736,
                'Sb Ka2 (2nd)': 0.949681
                }
        elif 'XRSHRXe_W' in kk+other_data:
            lines = {
                'Al_1': 2.72051,
                'Al_2': 2.72844,
                'Si_1': 2.70277
                }
            calib = {}
        elif 'XRSHRXe' in kk+other_data:
            lines = {
                '3D': 2.7204,
                '3F': 2.7290,
                'Na': 2.7364,
                'Mg': 2.7472
                }
            calib = {
                'Ti Ka1': 2.7490,
                'Ti Ka2': 2.7520,
                }
        elif 'XRSLR' in kk:
            lines = {
                'W: 3G': 1.49282,
                'W: 3D': 1.35881,
                }
            calib = {
                #'Cu Ka1': 1.540607,
                #'Cu Ka2': 1.544436,
                'Zn Ka1': 1.4351975,
                'Zn Ka2': 1.439042
            }

        # Loads Collection object
        coll = tf.data.load(coll_tf[ii])

        # Gets wavelength basis
        lamb_tf, _ = coll.get_diagnostic_lamb(
            kk,
            key_cam=key_cams[ii],
            lamb='lamb',
            ) # dim(nx, ny)
        dxi[kk]['lambda_A'] = lamb_tf*1e10 # [A]

        # Indexing
        if key_data == 'ToFu':
            indyx = int(dxi[kk]['npix'][1]/2-1)
        elif key_data == 'XICSRT':
            indyx = int(lamb_tf.shape[-1]/2-1)
        indxx = np.nanargmin(abs(
            lamb_tf[:,indyx] - lamb0[ii]*1e-10
            ))

        # Plotting
        fig4, ax4 = plt.subplots(2,2)

        tmp = np.sum(dxi[kk]['signal'], axis = 1) *scalex[kk]['h']
        ax4[0,0].plot(
            dxi[kk]['cents_cm'][0],
            tmp,
            '*-',
            label = key_data,
            color = color_data
            )

        if not norm_by_bin:
            ax4[0,0].fill_between(
                dxi[kk]['cents_cm'][0],
                tmp - np.sqrt(tmp),
                tmp + np.sqrt(tmp),
                color = color_data,
                alpha = 0.6
                )
            ax4[0,0].fill_between(
                dxi[kk]['cents_cm'][0],
                tmp - 2*np.sqrt(tmp),
                tmp + 2*np.sqrt(tmp),
                color = color_data,
                alpha = 0.3
                )

        for ll in lines.keys():
            indlx = np.nanargmin(abs(
                lamb_tf[:,indyx] - lines[ll]*1e-10
                ))

            ax4[0,0].text(
                dxi[kk]['cents_cm'][0][indlx],
                (np.sum(dxi[kk]['signal'], axis = 1) *scalex[kk]['h'])[indlx],
                ll,
                color = 'k'
                )
        for ll in calib.keys():
            indlx = np.nanargmin(abs(
                lamb_tf[:,indyx] - calib[ll]*1e-10
                ))
            ax4[0,0].axvspan(
                dxi[kk]['cents_cm'][0][indlx]-0.05,
                dxi[kk]['cents_cm'][0][indlx]+0.05,
                color = 'm',
                alpha = 0.3
                )
            ax4[0,0].text(
                dxi[kk]['cents_cm'][0][indlx],
                np.nanmax(np.sum(dxi[kk]['signal'], axis = 1) *scalex[kk]['h']),
                ll,
                color = 'k'
                )

        ax4[0,0].set_xlabel('horiz. bin [cm]')
        ax4[0,0].set_ylabel(label_h)
        ax4[0,0].yaxis.set_major_formatter(formatter)
        ax4[0,0].set_title('int. over all vert. bins')
        ax4[0,0].grid('on')
        #ax4[0,0].legend(labelcolor='linecolor')
        ax4[0,0].set_xlim(0.95*np.min(dxi[kk]['cents_cm'][0]), 1.05*np.max(dxi[kk]['cents_cm'][0]))

        tmp = dxi[kk]['signal'][:,indyx] *scalex[kk]['hv']
        ax4[1,0].plot(
            dxi[kk]['cents_cm'][0],
            tmp,
            '*-',
            label = key_data,
            color = color_data
            )

        if not norm_by_bin:
            ax4[1,0].fill_between(
                dxi[kk]['cents_cm'][0],
                tmp - np.sqrt(tmp),
                tmp + np.sqrt(tmp),
                color = color_data,
                alpha = 0.6
                )
            ax4[1,0].fill_between(
                dxi[kk]['cents_cm'][0],
                tmp - 2*np.sqrt(tmp),
                tmp + 2*np.sqrt(tmp),
                color = color_data,
                alpha = 0.3
                )

        ax4[1,0].set_xlabel('horiz. bin [cm]')
        ax4[1,0].set_ylabel(label_hv)
        ax4[1,0].yaxis.set_major_formatter(formatter)
        #ax4[1,0].set_title('vert. bin %i/%i'%(indy, dxi['npix'][1]-1))
        #ax4[1,0].set_title('vert. bin %i/%i'%(indyt, dtf['npix'][1]-1))
        ax4[1,0].set_title('vert. bin @ %0.2f cm'%(dxi[kk]['cents_cm'][1][indyx]))
        ax4[1,0].grid('on')
        ax4[1,0].set_xlim(0.95*np.min(dxi[kk]['cents_cm'][0]), 1.05*np.max(dxi[kk]['cents_cm'][0]))
        ax4[1,0].legend(labelcolor='linecolor')

        ax4[0,1].plot(
            np.sum(dxi[kk]['signal'], axis = 0) *scalex[kk]['v'],
            dxi[kk]['cents_cm'][1],
            '*-',
            color = color_data
            )

        if np.min(dxi[kk]['cents_cm'][1]) < 0:
            ff1 = 1.05
        else:
            ff1 = 0.95
        if np.max(dxi[kk]['cents_cm'][1]) < 0:
            ff2 = 0.95
        else:
            ff2 = 1.05

        ax4[0,1].set_ylabel('vert. bin [cm]')
        ax4[0,1].set_xlabel(label_v)
        ax4[0,1].xaxis.set_major_formatter(formatter)
        ax4[0,1].set_title('int. over all horz. bins')
        ax4[0,1].grid('on')
        ax4[0,1].set_ylim(ff1*np.min(dxi[kk]['cents_cm'][1]), ff2*np.max(dxi[kk]['cents_cm'][1]))

        ax4[1,1].plot(
            dxi[kk]['signal'][indxx,:] *scalex[kk]['hv'],
            dxi[kk]['cents_cm'][1],
            '*-',
            color = color_data
            )

        ax4[1,1].set_ylabel('vert. bin [cm]')
        ax4[1,1].set_xlabel(label_hv)
        ax4[1,1].xaxis.set_major_formatter(formatter)
        #ax4[1,1].set_title('horz. bin %i/%i'%(indx, dxi['npix'][0]-1))
        #ax4[1,1].set_title('horz. bin %i/%i'%(indxt, dtf['npix'][0]-1))
        ax4[1,1].set_title('horz. bin @ %0.2f cm'%(dxi[kk]['cents_cm'][0][indxx]))
        ax4[1,1].grid('on')
        ax4[1,1].set_ylim(ff1*np.min(dxi[kk]['cents_cm'][1]), ff2*np.max(dxi[kk]['cents_cm'][1]))

    # Output
    return dxi, scalex







