# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:03:00 2019
@author: ju-bar

Image shift utilities

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
from emilys.image.imagedata import maxpos, com
from emilys.image.peakfit import fit_local_gauss_2d
# %%
def get_rigid_shift(img, ref, method='pixel', maxshift=0.):
    '''
    Measures the rigid shift of data in the 2d array 'img' relative to 'ref'.

    Parameters
    ----------
        img : numpy.ndarray (M,N)
            test image
        ref : numpy.ndarray (M,N)
            reference image
        method : string, default: 'pixel'
            shift finding method
                'pixel' : closest pixel shift
                'com' : center of mass
                'peak' : peak fitting
        maxshift : float, default: 0. = no limit
            limitation of shift vector in pixels along both dimensions

    Returns
    -------
        numpy.ndarray (2,)
            rigid shift vector (x,y)
    '''
    lims = False # default correlation maximum search region is not limited
    ls = np.abs(maxshift) # check for search limit
    if (ls > 0.): lims = True # there is a search limit
    assert img.shape == ref.shape and len(img.shape) == 2, \
        'expecting 2d numpy arrays of equal shape as parameters 1 and 2'
    ndim = np.array(img.shape) # get image dimension
    csca = 1. / ndim[0] / ndim[1] # get rescaling for fft
    mi = np.mean(img) # get image mean
    si = np.std(img) # get image standard deviation
    mf = np.mean(ref) # get reference mean
    sf = np.std(ref) # get reference standard deviation
    fti = np.fft.fft2((img - mi)/si) # FT image
    ftr = np.fft.fft2((ref - mf)/sf) # FT reference
    iorg = (ndim / 2).astype(int) # get origin
    cfir0 = np.roll(np.real(np.fft.ifft2(fti * np.conjugate(ftr))), iorg, np.array([0,1])) * csca # cross-correlation rolled to origin
    dc = np.array(iorg)
    i00 = 0
    i10 = 0
    if lims: # extract part of the x-corr image to look for maximum
        ils = np.ceil(ls)
        i00 = int(max(0, iorg[0]-ils))
        i01 = int(min(ndim[0], iorg[0]+ils))
        i10 = int(max(0, iorg[1]-ils))
        i11 = int(min(ndim[1], iorg[1]+ils))
        dc = np.array(iorg) - np.array([i00,i10]) # move center
        cfir = cfir0[i00:i01+1,i10:i11+1].copy() # copy the data
    else:
        cfir = cfir0.copy() # copy all data
    if method == 'peak':
        p0 = maxpos(cfir) + np.array([i00,i10])
        peak_prm = fit_local_gauss_2d(cfir0, np.flip(p0), ls)
        fpos = np.array([peak_prm[0][0], peak_prm[0][1]]) - np.flip(iorg)
        ferr = np.array([peak_prm[1][0], peak_prm[1][1]])
        return [fpos, ferr]
    elif method == 'com':
        return [com(cfir) - np.flip(dc), np.array([0.2,0.2])] 
    return [maxpos(cfir) - np.flip(dc), np.array([0.5,0.5])]

#%%
