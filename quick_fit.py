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
from utils import biweight_location, biweight_midvariance, biweight_bin
from astroquery.vizier import Vizier 
import astropy.units as u 
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord

def parse_args(argv=None):
    # Arguments to parse include ssp code, metallicity, isochrone choice, 
    #   whether this is real data or mock data
    parser = ap.ArgumentParser(description="stellarSEDfit",
                            formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument("-f","--filename", 
                        help='''File to be read for star photometry''',
                        type=str, default=None)

    parser.add_argument("-o","--outfolder", 
                        help='''Folder to write output to.''',
                        type=str, default=None)
                        
    parser.add_argument("-e","--ebv",
                        help='''Extinction, e(b-v)=0.02, for star field''',
                        type=float, default=0.02)
                        
    parser.add_argument("-p","--make_plot",
                        help='''If you want to make plots, 
                        just have this option''',
                        action="count", default=0)

    parser.add_argument("-wi","--wave_init",
                        help='''Initial wavelength for bin, default=3540''',
                        type=float, default=3540)

    parser.add_argument("-wf","--wave_final",
                        help='''Final wavelength for bin, default=5540''',
                        type=float, default=5540)

    parser.add_argument("-bs","--bin_size",
                        help='''Bin size for wavelength, default=100''',
                        type=float, default=100)
                        
    args = parser.parse_args(args=argv)
    
    return args

def Cardelli(wv,Rv=None):
    if Rv is None:
        Rv = 3.1  # setting default Rv value
        
    sel = np.logical_and(1.0/(wv/10000.0)<3.3,1.0/(wv/10000.0)>1.1)
    m   = 1.0/(wv[sel]/10000.0)
    y   = (m-1.82)
    a   = 1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 \
    + 0.72085 * y**4 + 0.01979 * y**5 - 0.77530 * y**6 +0.32999 * y**7
    b   = 1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 \
    - 5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6 -2.09002 * y**7
    z1  = a + b / Rv;
    
    sel1 = np.logical_and(1.0/(wv/10000)>=3.3,1.0/(wv/10000.0)<5.9)
    m    = 1.0/(wv[sel1]/10000)
    Fa   = 0
    Fb   = 0
    a    = 1.752-0.316*m - 0.104/((m-4.67)**2 + 0.341) + Fa
    b    = -3.090+1.825*m+1.206/((m-4.62)**2 + 0.263) + Fb;
    z2   = a + b / Rv;
    
    sel2 = 1.0/(wv/10000.0)>=5.9
    m    = 1.0/(wv[sel2]/10000.0)
    Fa   = -.04473*(m-5.9)**2 - 0.009779*(m-5.9)**3
    Fb   = 0.2130*(m-5.9)**2 +0.1207*(m-5.9)**3
    a    = 1.752-0.316*m - 0.104/((m-4.67)**2 + 0.341) + Fa
    b    = -3.090+1.825*m+1.206/((m-4.62)**2 + 0.263) + Fb
    z3   = a + b / Rv;
    
    sel3 = 1.0/(wv/10000.0)<1.1
    m    = 1.0/(wv[sel3]/10000.0)
    a    = 0.574*m**1.61
    b    = -.527*m**1.61
    z4   = a + b / Rv
    
    z       = np.zeros((np.sum(sel)+np.sum(sel1)+np.sum(sel2)+np.sum(sel3),))
    z[sel]  = z1
    z[sel1] = z2
    z[sel2] = z3
    z[sel3] = z4
    return z * Rv

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
 
def make_plot(vmax, errbounds, stargrid, lnp, ind, normspec,
              wave, avgspec, stdspec, wv_vector, sh, Id, m, chi):
    fig = plt.figure(figsize=(8,6))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    ax1.set_position([0.15,0.35,0.7,0.5])
    ax2.set_position([0.15,0.15,0.7,0.2])
    vmin = vmax - errbounds
    sc = ax1.scatter(stargrid[:,2],stargrid[:,3],c=lnp, vmin=vmin, 
                 vmax=vmax)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top') 
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax, orientation='vertical')
    for i,norms in zip(ind,normspec):
        ax2.plot(wave, norms, color = sc.cmap((lnp[i]-vmin)
                                              /(vmax-vmin)))
        ax2.plot(wave,avgspec,'r-', lw=2)
        ax2.plot(wave,avgspec-stdspec,'r--',lw=1.5)
        ax2.plot(wave,avgspec+stdspec,'r--',lw=1.5)
        s = np.array([stargrid[i,4]-stargrid[i,5] + m[1], m[1]])
        ax2.scatter(wv_vector[0:2],10**(-0.4*(s-23.9))/1e29*3e18
                                    /wv_vector[0:2]**2,
                    c=[lnp[i],lnp[i]],vmin=vmin, vmax=vmax)
    ax2.scatter(wv_vector[0:2],10**(-0.4*(m-23.9))/1e29*3e18
                              /wv_vector[0:2]**2,color='r',marker='x')
    ax1.text(3.6,3,r'${\chi}^2 =$ %0.2f' %chi)
    ax2.set_xlabel('Wavelength')
    ax2.set_ylabel(r'F$_{\lambda}$')
    ax1.set_xlabel('Log Temp')
    ax1.set_ylabel('Log g')
    plt.savefig(op.join('plots','%06d_%i_prob.png' %(sh,Id)))
    plt.close()


def queryGAIA2(ra, dec, boxsize, maxsources=10000):
    """
    Queries USNO_A2.
    ra  = center RA of field
    dec = center DEC of field
    radius = determines size of radius around ra/dec for which sources should
              be retrieved
    return array of stars with format
    IDa IDb RA DEC 0. B R 0. 0.
    """

    vquery = Vizier(columns=['Source', 'RA_ICRS', 'DE_ICRS',
                             'phot_g_mean_mag', '_RAJ2000', '_DEJ2000',
                             'pmRA', 'pmDE', 'Plx', 'e_Plx'],
                    row_limit = maxsources) 
 
    field = coord.SkyCoord(ra=ra, dec=dec, 
                           unit=(u.deg, u.deg), 
                           frame='fk5')
    D = vquery.query_region(field, 
                               width=("%fd" % boxsize), 
                               catalog="I/345/gaia2")
    try:
        Data =  D[0] 
    except:
        return np.array([])
    oo = []
    g_i = 1.  # g-i color
    for i, obj in enumerate(Data['Source']):
        oid_a = Data['Source'][i]
        oid_b = 0
        ra = Data['_RAJ2000'][i]
        dec = Data['_DEJ2000'][i]
        pmra = Data['pmRA'][i]
        pmde = Data['pmDE'][i]
        G = Data['Gmag'][i]
        g = G + 0.0939 + 0.6758 * g_i + 0.04 * g_i**2 - 0.003 * g_i**3
        if np.any([j for j in Data.mask[i]]):
            continue
        oo.append([oid_a, oid_b, ra, dec, 0., g, 0., 0., 0., pmra, pmde])
    return np.array(oo)   


def main():
    args = parse_args()
    # Load Star Grid and star names (different data type, and this is a sol(n))
    stargrid = np.loadtxt('stargrid-150501.dat',usecols=[1,2,3,4,8,9,10,11,12],
                          skiprows=1)
    starnames = np.loadtxt('stargrid-150501.dat',usecols=[0],
                           skiprows=1,dtype=str)
    # Get MILES wavelength array
    p = fits.open(op.join('miles_spec','S'+starnames[0]+'.fits'))
    waveo = (p[0].header['CRVAL1'] + np.linspace(0,len(p[0].data)-1,
                                                len(p[0].data))
                                                *p[0].header['CDELT1'])
    # Load MILES spectra                                            
    spectra = load_spectra(waveo, stargrid[:,0], starnames)
    
    # define wavelength 
    wave = np.arange(args.wave_init,args.wave_final+args.bin_size,
                     args.bin_size,dtype=float)
    
    # load the data from file
    data = np.loadtxt(args.filename)
    shot, ID = np.loadtxt(args.filename, usecols=[0,1],dtype=int,unpack=True)
    
    # Load the priors on Mg and Z
    P = load_prior()
    
    # In case the "plots" directory does not exist
    mkpath('plots')
    mkpath('output')
    
    # Columns from data and stargrid for u,g,r,i (z is a +1 in loop)
    cols = [4,5,6,7]
    
    # Extinction and wavelength vector for ugriz
    ext_vector = np.array([4.892,3.771,2.723,2.090,1.500])
    wv_vector  = np.array([3556,4702,6175,7489,8946])
    
    # Guessing at an error vector from modeling and photometry (no erros provided)
    mod_err = .02**2 + .02**2
    e1 = np.sqrt(.05**2 + .02**2 + mod_err) # u-g errors
    e2 = np.sqrt(.02**2 + .02**2 + mod_err) # g-r, r-i, i-z errors
    err_vector = np.array([e1,e2,e2,e2])
    
    # Using input extinction and correcting magnitudes
    ebv = args.ebv
    extinction = Cardelli(wave)
    stargrid[:,4:9] = stargrid[:,4:9] + ext_vector*ebv
    
    # remove an odd point from the grid
    sel = ((stargrid[:,2]>3.76)*(stargrid[:,2]<3.79)
                       *(stargrid[:,3]>2.5)*(stargrid[:,3]<3.00))
    stargrid = stargrid[~sel,:]
    spectra = spectra[~sel,:]
    
    # Calculate color distance chi2 and use for likelihood
    d = np.zeros((len(data),len(stargrid),4))
    for i,col in enumerate(cols):
        d[:,:,i]=1/err_vector[i]*((data[:,col]-data[:,col+1])[:,np.newaxis] 
                                     -(stargrid[:,col]- stargrid[:,col+1]))
    dd = d**2 + np.log(2*np.pi*err_vector**2)
    lnlike = -0.5 * dd.sum(axis=2)
    chi2 = 1./(len(err_vector)+1)*(d**2).sum(axis=2)
    # Calculate prior and add to likelihood for probability
    lnprior = P(stargrid[:,0],stargrid[:,1])
    lnprob = lnlike + lnprior
    
    # Loop through all sources to best fit spectra with errors
    for lnp,sh,Id,m,chi in zip(lnprob,shot,ID,data[:,4:6],chi2):
        bv = np.argsort(lnp)[-3]
        vmax=lnp[bv]
        errbounds = 2.5
        ind = np.where((vmax - lnp)<errbounds)[0]
        normspec = []
        for i in ind:
            fac = 10**(-0.4*(m[1] - (stargrid[i,5]-ext_vector[1]*ebv)))
            normspec.append(fac * biweight_bin(wave,waveo,spectra[i])
                            *10**(-.4*ebv*extinction))
        if len(normspec)>2:
            avgspec = biweight_location(normspec,axis=(0,))
            mn = np.min(normspec, axis=0)
            mx = np.max(normspec, axis=0)
            stdspec = mx/2. - mn/2.
        else:
            avgspec = np.mean(normspec, axis=(0,))
            stdspec = 0.2 * avgspec
        if args.outfolder is None:
            args.outfolder = 'output'
        F = np.array([wave,avgspec,stdspec],dtype='float32').swapaxes(0,1)
        n,d = F.shape
        F1 = np.zeros((n+1,d))
        F1[1:,:] = F
        F1[0,:] = [chi[bv],0,0]
        np.savetxt(op.join(args.outfolder,'%06d_%i.txt' %(sh, Id)), F1)
        if args.make_plot:
            make_plot(vmax, errbounds, stargrid, lnp, ind, normspec,
                      wave, avgspec, stdspec, wv_vector, sh, Id, m, chi[bv])
              
if __name__=='__main__':
    main()