'''

Functions to calculate histograms for XICSRT

cjperks
Aug 5, 2024

'''

# Modules
import numpy as np

import cmp_tofu_xicsrt.utils._conv as cv

__all__ = [
    '_calc_ang_hist',
    '_calc_det_hist'
    ]

# Calculates the angular distribution of rays from XICSRT
def _calc_ang_hist(
    data = None,
    sim_type = 'mv',
    config = None,
    plt_all = True,
    # Rocking curve data
    material = 'Germanium',
    miller = np.r_[2., 0., 2.],
    lamb0 = 1.61,
    # Binning controls
    nbins = 50,
    ):

    xi_src = np.empty((0,3)) # dim(nfound, 3)
    xi_cry = np.empty((0,3)) # dim(nfound, 3)
    xi_det = np.empty((0,3)) # dim(nfound, 3)

    xi_vec_s2c = np.empty((0,3)) # dim(nfound, 3)
    xi_vec_c2d = np.empty((0,3)) # dim(nfound, 3)

    if sim_type == 'mv':
        for key in data.keys():
            xi_src = np.vstack((
                cv._xicsrt2tofu(
                    data = data[key]['XICSRT']['source']['origin']
                    ),
                xi_src
                ))
            xi_cry = np.vstack((
                cv._xicsrt2tofu(
                    data = data[key]['XICSRT']['crystal']['origin']
                    ),
                xi_cry
                ))
            xi_det = np.vstack((
                cv._xicsrt2tofu(
                    data = data[key]['XICSRT']['detector']['origin']
                    ),
                xi_det
                ))

            xi_vec_s2c = np.vstack((
                cv._xicsrt2tofu(
                    data = data[key]['XICSRT']['source']['direction']
                    ),
                xi_vec_s2c
                ))
            xi_vec_c2d = np.vstack((
                cv._xicsrt2tofu(
                    data = data[key]['XICSRT']['crystal']['direction']
                    ),
                xi_vec_c2d
                ))

    # Calculates the crystal normal vector
    xi_cry_norm = xi_vec_c2d - xi_vec_s2c
    xi_cry_norm /= np.linalg.norm(xi_cry_norm, axis=1)[:,None]

    # Calculates the ingoing/outgoing angles
    xi_ang_in = abs(90- np.arccos(
        np.einsum('ij,ij->i',
            xi_vec_s2c,
            xi_cry_norm
            )
        )*180/np.pi)
    xi_ang_out = abs(90- np.arccos(
        np.einsum('ij,ij->i',
            xi_vec_c2d,
            xi_cry_norm
            )
        )*180/np.pi)


    bragg = 180/np.pi * np.arcsin(
        lamb0/2
        /config['optics']['crystal']['crystal_spacing']
        )

    # Histograms the angules
    hist_in, bins_in = np.histogram(
        xi_ang_in-bragg,
        bins = nbins
        )
    hist_out, bins_out = np.histogram(
        xi_ang_out-bragg,
        bins = nbins
        )

    # Plotting
    if plt_all:
        from atomic_world.run_TOFU.main import get_rocking as gr
        drock = gr._get_rocking(
            material = material,
            miller = miller,
            lamb0 = lamb0,
            plt_all = False
            )
        
        fig1, ax1 = plt.subplots(1,2)

        ax1[0].bar(
            (bins_in[1:]+bins_in[:-1])/2,
            hist_in/np.max(hist_in),
            width = np.diff(bins_in)
            )
        ax1[0].plot(
            (drock['angle']-drock['bragg'])*180/np.pi,
            drock['pwr']/np.max(drock['pwr']),
            'r-'
            )

        ax1[0].set_xlabel('incident angle - bragg [deg]')
        ax1[0].set_ylabel('norm. dist')
        ax1[0].set_title('angle in')

        ax1[1].bar(
            (bins_out[1:]+bins_out[:-1])/2,
            hist_out/np.max(hist_out),
            width = np.diff(bins_out),
            label = 'angle dist.'
            )
        ax1[1].plot(
            (drock['angle']-drock['bragg'])*180/np.pi,
            drock['pwr']/np.max(drock['pwr']),
            'r-',
            label = 'rocking curve'
            )
        ax1[1].legend(labelcolor='linecolor')

        ax1[1].set_xlabel('incident angle - bragg [deg]')
        ax1[1].set_ylabel('norm. dist')
        ax1[1].set_title('angle out')

        fig1.suptitle('XICSRT', color = 'blue')

        intr = np.trapz(drock['pwr'], drock['angle']*1e6)
        wid = np.diff(bins_in)*np.pi/180*1e6 # [urad]
        int_in = np.sum(hist_in * wid/np.max(hist_in)*np.max(drock['pwr']))
        wid = np.diff(bins_out)*np.pi/180*1e6 # [urad]
        int_out = np.sum(hist_out *wid/np.max(hist_out)*np.max(drock['pwr']))

        print('Integrated Reflectivity')
        print('Rocking curve [urad]')
        print(intr)
        print('Angle in')
        print(int_in)
        print('Error [%]')
        print(abs(int_in/intr-1)*100)
        print('Angle out [urad]')
        print(int_out)
        print('Error [%]')
        print(abs(int_out/intr-1)*100)
        

    # Output
    return {
        'hist_in': hist_in,
        'ang_in': bins_in,
        'hist_out': hist_out,
        'ang_out': bins_out
        }

# Calculates histogram of the detector surface
def _calc_det_hist(
    rays = None,
    config = None,
    nx = 512,
    ny = 256,
    ):
    '''
    *assumes quantities are in XICSRT's coordinate representation
    '''

    # Detector geometry
    zvect = config['optics']['detector']['zaxis']
    xvect = config['optics']['detector']['xaxis']
    yvect = np.cross(zvect, xvect)
    cent = config['optics']['detector']['origin']
    dx = config['optics']['detector']['xsize']/2
    dy = config['optics']['detector']['ysize']/2

    # Rays origin in frame of detector
    xs = np.dot(rays-cent, xvect) # [m]
    ys = np.dot(rays-cent, yvect) # [m]

    # Counts per bin
    ncnt, xbin, ybin = np.histogram2d(    
        xs, ys,
        bins=[nx, ny],
        range=[[-dx,dx], [-dy,dy]]
        ) 

    # Output, dim(hor_pix, vert_pix)
    return {
        'counts':ncnt, 
        'xbin':xbin, 
        'ybin':ybin, 
        'aspect':dx/dy
        }
