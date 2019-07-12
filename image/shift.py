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
from emilys.image.imagedata import maxpos
# %%
def get_rigid_shift(img, ref, method='pixel', maxshift=0.):
    '''
    Measures the rigid shift of data in the 2d array 'img' relative to 'ref'.

    Parameters:
        img : numpy.ndarray (M,N)
            test image
        ref : numpy.ndarray (M,N)
            reference image
        method : string, default: 'pixel'
            shift finding method
                'pixel' : closest pixel shift
        maxshift : float, default: 0. = no limit
            limitation of shift vector in pixels along both dimensions

    Return:
        numpy.ndarray (2,)
            rigid shift vector (x,y)
    '''
    lims = False
    ls = np.abs(maxshift)
    if (ls > 0.): lims = True
    assert img.shape == ref.shape and len(img.shape) == 2, \
        'expecting 2d numpy arrays of equal shape as parameters 1 and 2'
    ndim = np.array(img.shape)
    csca = 1. / ndim[0] / ndim[1]
    mi = np.mean(img)
    si = np.std(img)
    mf = np.mean(ref)
    sf = np.std(ref)
    fti = np.fft.fft2((img - mi)/si)
    ftr = np.fft.fft2((ref - mf)/sf)
    iorg = (ndim / 2).astype(int)
    cfir0 = np.roll(np.real(np.fft.ifft2(fti * np.conjugate(ftr))), iorg, np.array([0,1])) * csca
    dc = iorg
    if lims:
        ils = np.ceil(ls)
        i00 = int(max(0, iorg[0]-ils))
        i01 = int(min(ndim[0], iorg[0]+ils))
        i10 = int(max(0, iorg[1]-ils))
        i11 = int(min(ndim[1], iorg[1]+ils))
        dc = iorg - np.array([i00,i10])
        cfir = cfir0[i00:i01+1,i10:i11+1].copy()
    else:
        cfir = cfir0.copy()
    method_switch = {
        'pixel' : maxpos(cfir) - np.flip(dc)
    }
    return method_switch.get(method, maxpos(cfir) - np.flip(dc))
    


#%%
