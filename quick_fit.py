# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:01:47 2017

@author: gregz
"""

import numpy as np
import argparse as ap
import os.path as op
import matplotlib.pyplot as plt
import scipy.interpolate as scint
from distutils.dir_util import mkpath
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable


def parse_args(argv=None):
    # Arguments to parse include ssp code, metallicity, isochrone choice, 
    #   whether this is real data or mock data
    parser = ap.ArgumentParser(description="stellarSEDfit",
                            formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument("-f","--filename", 
                        help='''File to be read for star photometry''',
                        type=str, default=None)
                        
    parser.add_argument("-e","--ebv",
                        help='''Extinction, e(b-v)=0.02, for star field''',
                        type=float, default=0.02)
                        
    args = parser.parse_args(args=argv)
    
    return args

def load_prior():
    xd  = np.loadtxt('mgvz_mg_x.dat')
    yd  = np.loadtxt('mgvz_zmet_y.dat')
    Z   = np.loadtxt('mgvz_prior_z.dat')
    X,Y = np.meshgrid(xd,yd)
    a,b = np.shape(Z)
    zv  = np.reshape(Z,a*b)
    xv  = np.reshape(X,a*b)
    yv  = np.reshape(Y,a*b)
    P   = scint.LinearNDInterpolator(np.array(zip(xv,yv)),zv)
    return P

def load_spectra(wave, Mg, starnames):
    fn = op.join('miles_spec','all_spec.fits')
    if op.exists(fn):
        spec = fits.open(fn)[0].data
    else:    
        f = []
        for star in starnames:
            hdu = fits.open(op.join('miles_spec','S'+star+'.fits'))
            hdu_data = hdu[0].data
            f.append(hdu_data*1.)
            del hdu_data
            del hdu[0].data
        hdu1 = fits.open(op.join('miles_spec','S'+star+'.fits'))
        F = np.array(f)
        gfilter = np.loadtxt('sloan_g')
        G=scint.interp1d(gfilter[:,0], gfilter[:,1], bounds_error=False, 
                         fill_value=0.0)
        Rl = G(wave)
        abstd = 3631/(3.34e4*wave**2)
        spec = np.zeros(F.shape,dtype='float32')
        for i,tflux in enumerate(F):
            a = np.sum(tflux * Rl * wave)
            b = np.sum(abstd * Rl * wave)
            m = -2.5 * np.log10(a/b)
            fac = 10**(-0.4*(Mg[i]-m))
            spec[i,:] = tflux * fac
        hdu1 = fits.PrimaryHDU(spec,header=hdu[0].header)
        hdu1.writeto(fn,overwrite=True)
    return spec
    
def main():
    args = parse_args()
    stargrid = np.loadtxt('stargrid-150501.dat',usecols=[1,2,3,4,8,9,10,11,12],
                          skiprows=1)
    starnames = np.loadtxt('stargrid-150501.dat',usecols=[0],
                           skiprows=1,dtype=str)
    p = fits.open(op.join('miles_spec','S'+starnames[0]+'.fits'))
    waveo = (p[0].header['CRVAL1'] + np.linspace(0,len(p[0].data)-1,
                                                len(p[0].data))
                                                *p[0].header['CDELT1'])
    spectra = load_spectra(waveo, stargrid[:,0], starnames)
    
    wave = np.linspace(3500,5500,100)
    data = np.loadtxt(args.filename)
    shot, ID = np.loadtxt(args.filename, usecols=[0,1],dtype=int,unpack=True)
    P = load_prior()
    mkpath('plots')
    cols = [4,5,6,7]
    ext_vector = np.array([4.892,3.771,2.723,2.090,1.500])
    mod_err = .02**2 + .02**2
    e1 = np.sqrt(.05**2 + .02**2 + mod_err) # u-g errors
    e2 = np.sqrt(.02**2 + .02**2 + mod_err) # g-r, r-i, i-z errors
    err_vector = np.array([e1,e2,e2,e2])
    ebv = args.ebv
    d = np.zeros((len(data),len(stargrid),4))
    for i,col in enumerate(cols):
        d[:,:,i]=1/err_vector[i]*((data[:,col]-data[:,col+1])[:,np.newaxis] 
                                     -((stargrid[:,col]+ext_vector[i]*ebv)
                                   -(stargrid[:,col+1]+ext_vector[i+1]*ebv)))
    dd = d**2 + np.log(2*np.pi*err_vector**2)
    lnlike = -0.5 * dd.sum(axis=2)
    lnprior = P(stargrid[:,0],stargrid[:,1])
    lnprob = lnlike + lnprior
    
    for lnp,sh,Id,mg in zip(lnprob,shot,ID,data[:,5]):
        print(sh, Id)
        vmax=np.sort(lnp)[-2]
        fig = plt.figure(figsize=(8,6))
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        ax1.set_position([0.15,0.35,0.7,0.5])
        ax2.set_position([0.15,0.15,0.7,0.2])
        sc = ax1.scatter(stargrid[:,2],stargrid[:,3],c=lnp, vmin=vmax-7.5, 
                         vmax=vmax)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top') 
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sc, cax=cax, orientation='vertical')
        ind = np.where((vmax - lnp)<2.5)[0]
        for i in ind:
            fac = 10**(-0.4*(mg - stargrid[i,0]))
            ax2.plot(wave, fac * np.interp(wave,waveo,spectra[i]))
        plt.savefig(op.join('plots','%06d_%i_prob.png' %(sh,Id)))
        plt.close()
if __name__=='__main__':
    main()