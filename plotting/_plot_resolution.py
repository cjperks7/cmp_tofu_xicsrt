'''

Script to process and plot characterizing the spectral-/spatial-resolution
of a spectrometer

cjperks
Nov 9, 2024

'''

# Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 16})

__all__ = [
    'plt_res'
    ]

###############################################
#
#           Main
#
###############################################

# Plots spectral-/spatial-resolution data
def plt_res(
    dout = None,
    ):

    # Loads data
    if isinstance(dout, str):
        dout = np.load(dout, allow_pickle=True)['arr_0'][()]

    # Prepares output data
    ddata = _prep_data(
        dout=dout,
        )
    dxi = ddata['XICSRT']
    dtf = ddata['ToFu']

    xind_xi = dxi['mid']['xind']
    yind_xi = dxi['mid']['yind']
    xind_tf = dtf['mid']['xind']
    yind_tf = dtf['mid']['yind']

    # Rescale ToFu if different pixel binning
    dx_xi = np.mean(abs(
        dout['XICSRT']['cents_cm'][0][1:] - dout['XICSRT']['cents_cm'][0][:-1]
        ))
    dy_xi = np.mean(abs(
        dout['XICSRT']['cents_cm'][1][1:] - dout['XICSRT']['cents_cm'][1][:-1]
        ))
    dx_tf = np.mean(abs(
        dout['ToFu']['cents_cm'][0][1:] - dout['ToFu']['cents_cm'][0][:-1]
        ))
    dy_tf = np.mean(abs(
        dout['ToFu']['cents_cm'][1][1:] - dout['ToFu']['cents_cm'][1][:-1]
        ))
    scalet_0 = (
        dx_tf/dx_xi
        *dy_tf/dy_tf
        )
    scalet_1 = (
        dy_tf/dy_xi
        )

    #### --- Plots wavelength scan --- ####

    figy = plt.figure(figsize = (16,8))
    gsy = gridspec.GridSpec(2,1, hspace = 0.4)

    # Plots XICSRT
    axy = figy.add_subplot(gsy[0,0])
    for xx in np.arange(dxi['lamb_scan']['data'].shape[0]): # loop over horiz. pixels
        lab = int(xx - xind_xi)

        if np.sum(dxi['lamb_scan']['data'][xx,yind_xi,:].flatten()) >1e-16:
            axy.plot(
                dxi['lamb_scan']['lambda_AA'],
                dxi['lamb_scan']['data'][xx,yind_xi,:],
                '*-',
                label = 'horz. bin %i'%(lab)
                )
    
    axy.legend(labelcolor='linecolor')
    axy.grid('on')
    axy.set_xlabel(r'$\lambda$ [$\AA$]')
    axy.set_ylabel(r'signal [#ph/bin$^2$]')
    axy.set_title(
        'XICSRT: vert. bin %i/%i'%(yind_xi, dxi['lamb_scan']['data'].shape[1]),
        color = 'b'
        )

    # Plots ToFu
    axy = figy.add_subplot(gsy[1,0])
    for xx in np.arange(dtf['lamb_scan']['data'].shape[0]): # loop over horiz. pixels
        lab = int(xx - xind_tf)

        if np.sum(dtf['lamb_scan']['data'][xx,yind_tf,:].flatten())*scalet_0 >1e-16:
            axy.plot(
                dtf['lamb_scan']['lambda_AA'],
                dtf['lamb_scan']['data'][xx,yind_tf,:]*scalet_0,
                '*-',
                label = 'horz. bin %i'%(lab)
                )
    
    axy.legend(labelcolor='linecolor')
    axy.grid('on')
    axy.set_xlabel(r'$\lambda$ [$\AA$]')
    axy.set_ylabel(r'signal [#ph/bin$^2$]')
    axy.set_title(
        'ToFu: vert. bin %i/%i'%(yind_tf, dtf['lamb_scan']['data'].shape[1]),
        color =  'r'
        )
    
     #### --- Plots Z scan --- ####

    figz = plt.figure(figsize = (16,8))
    gsz = gridspec.GridSpec(2,1, hspace = 0.4)

    # Plots XICSRT
    axz = figz.add_subplot(gsz[0,0])
    for yy in np.arange(dxi['Z_scan']['data'].shape[1]): # loop over vert. pixels
        lab = int(yy - yind_xi)

        if np.sum(dxi['Z_scan']['data'][xind_xi, yy,:].flatten()) >1e-16:
            axz.plot(
                dxi['Z_scan']['pt'][:,-1]*100,
                dxi['Z_scan']['data'][xind_xi,yy,:],
                '*-',
                label = 'vert. bin %i'%(lab)
                )
    
    axz.legend(labelcolor='linecolor')
    axz.grid('on')
    axz.set_xlabel(r'Z [cm]')
    axz.set_ylabel(r'signal [#ph/bin$^2$]')
    axz.set_title(
        'XICSRT; horz. bin %i/%i'%(xind_xi, dxi['Z_scan']['data'].shape[0]),
        color = 'b'
        )

    # Plots ToFu
    axz = figz.add_subplot(gsz[1,0])
    for yy in np.arange(dtf['Z_scan']['data'].shape[1]): # loop over vert. pixels
        lab = int(yy - yind_tf)

        if np.sum(dtf['Z_scan']['data'][xind_tf,yy,:].flatten())*scalet_0 >1e-16:
            axz.plot(
                dtf['Z_scan']['pt'][:,-1]*100,
                dtf['Z_scan']['data'][xind_tf,yy,:]*scalet_0,
                '*-',
                label = 'vert. bin %i'%(lab)
                )
    
    axz.legend(labelcolor='linecolor')
    axz.grid('on')
    axz.set_xlabel(r'Z [cm]')
    axz.set_ylabel(r'signal [#ph/bin$^2$]')
    axz.set_title(
        'ToFu; horz. bin %i/%i'%(xind_tf, dtf['Z_scan']['data'].shape[0]),
        color = 'r'
        )


###############################################
#
#           Utils
#
###############################################

# Organizes output data
def _prep_data(
    dout=None,
    ):

    # Init
    ddata = {}
    ddata['XICSRT'] = {}
    ddata['ToFu'] = {}
    oxi = ddata['XICSRT']
    otf = ddata['ToFu']

    dxi = dout['XICSRT']
    dtf = dout['ToFu']

    # Number of Z/lambda points
    nZ = sum(1 for key in dxi if key.startswith('zind'))
    ny = sum(1 for key in dxi['zind_0'] if key.startswith('yind'))

    # Mid points
    midZ = int((nZ-1)/2)
    midy = int((ny-1)/2)

    # Organizes data scanning over Z at fixed wavelength
    oxi['Z_scan'] = {}
    oxi['Z_scan']['data'] = np.zeros(dxi['zind_0']['yind_0']['signal'].shape + (nZ,)) # dim(horz., vert., nZ)
    oxi['Z_scan']['pt'] = np.zeros((nZ,)+(3,)) # dim(nZ,3)

    for zz in np.arange(nZ):
        oxi['Z_scan']['data'][:,:,zz] = dxi['zind_%i'%(zz)]['yind_%i'%(midy)]['signal']
        oxi['Z_scan']['pt'][zz,:] = dxi['zind_%i'%(zz)]['pt']
    oxi['Z_scan']['lambda_AA'] = dxi['zind_%i'%(zz)]['yind_%i'%(midy)]['lamb_AA']

    otf['Z_scan'] = {}
    otf['Z_scan']['data'] = np.zeros(dtf['zind_0']['yind_0']['signal'].shape + (nZ,)) # dim(horz., vert., nZ)
    otf['Z_scan']['pt'] = np.zeros((nZ,)+(3,)) # dim(nZ,3)

    for zz in np.arange(nZ):
        otf['Z_scan']['data'][:,:,zz] = dtf['zind_%i'%(zz)]['yind_%i'%(midy)]['signal']
        otf['Z_scan']['pt'][zz,:] = dtf['zind_%i'%(zz)]['pt']
    otf['Z_scan']['lambda_AA'] = dtf['zind_%i'%(zz)]['yind_%i'%(midy)]['lamb_AA']

    # Organizes data scanning over wavelength at fixed Z
    oxi['lamb_scan'] = {}
    oxi['lamb_scan']['data'] = np.zeros(dxi['zind_0']['yind_0']['signal'].shape + (ny,)) # dim(horz., vert., nlambda)
    oxi['lamb_scan']['lambda_AA'] = np.zeros(ny)

    for yy in np.arange(ny):
        oxi['lamb_scan']['data'][:,:,yy] = dxi['zind_%i'%(midZ)]['yind_%i'%(yy)]['signal']
        oxi['lamb_scan']['lambda_AA'][yy] = dxi['zind_%i'%(midZ)]['yind_%i'%(yy)]['lamb_AA']
    oxi['lamb_scan']['pt'] = dxi['zind_%i'%(midZ)]['pt']

    otf['lamb_scan'] = {}
    otf['lamb_scan']['data'] = np.zeros(dtf['zind_0']['yind_0']['signal'].shape + (ny,)) # dim(horz., vert., nlambda)
    otf['lamb_scan']['lambda_AA'] = np.zeros(ny)

    for yy in np.arange(ny):
        otf['lamb_scan']['data'][:,:,yy] = dtf['zind_%i'%(midZ)]['yind_%i'%(yy)]['signal']
        otf['lamb_scan']['lambda_AA'][yy] = dtf['zind_%i'%(midZ)]['yind_%i'%(yy)]['lamb_AA']
    otf['lamb_scan']['pt'] = dtf['zind_%i'%(midZ)]['pt']

    # Finds brightest pixel about mid-point of scan
    oxi['mid'] = {}
    oxi['mid']['yind'] = np.argmax(
        np.sum(dxi['zind_%i'%(midZ)]['yind_%i'%(midy)]['signal'], axis = 0)
        )
    oxi['mid']['xind'] = np.argmax(
        np.sum(dxi['zind_%i'%(midZ)]['yind_%i'%(midy)]['signal'], axis = 1)
        )

    otf['mid'] = {}
    otf['mid']['yind'] = np.argmax(
        np.sum(dtf['zind_%i'%(midZ)]['yind_%i'%(midy)]['signal'], axis = 0)
        )
    otf['mid']['xind'] = np.argmax(
        np.sum(dtf['zind_%i'%(midZ)]['yind_%i'%(midy)]['signal'], axis = 1)
        )

    # Output
    return ddata

