# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:57:00 2019
@author: ju-bar

Peak fitting

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
from scipy.optimize import curve_fit
from emilys.functions.peaks import gauss_2d
# %%
def get_data_in_circroi(image, pos, rad, wrap=True):
    '''
    Extracts data points in a circular region of interest from an image.

    Parameters:
        image : numpy.ndarray
            image data
        pos : array (x,y)
            center position for the fit region of interest
        rad : float
            radius of the fit region of interest
            must be larger than 2
        wrap : bool
            flag for using periodic wrap around when crossing
            input image bounds

    Return:
        [number of image points n, positions, image values]
        [int, numpy.ndarray (n,2), numpy.ndarray (n)]
    '''
    ndim = np.array(image.shape)
    assert ndim.size == 2, 'expecting a 2d array as parameter 1'
    assert np.size(pos) == 2, 'expecting a tuple as parameter 2'
    assert np.abs(rad) > 2, 'expecting rad > 2 (parameter 3)'
    icent = np.array(pos).astype(int)
    irad = np.ceil(np.abs(rad)) + 1
    ir0 = icent - np.array([irad,irad], dtype=int)
    ir1 = icent + np.array([irad,irad], dtype=int)
    ndat = (ir1[0] - ir1[0] + 1) * (ir1[1] - ir1[1] + 1)
    datx = np.zeros((ndat,2), dtype=float)
    daty = np.zeros(ndat, dtype=float)
    nfit = 0
    for k in range(ir0[1], ir1[1]+1):
        ik = k
        if wrap:
            ik = int( k%ndim[0] )
        else:
            if (k < 0 or k >= ndim[0]):
                continue # skip this row
        dy = k - pos[1]
        dy2 = dy**2
        for h in range(ir0[0], ir1[1]+1):
            ih = h
            if wrap:
                ih = int( h%ndim[1] )
            else:
                if (h < 0 or h >= ndim[1]):
                    continue # skip this columns
            dx = h - pos[0]
            d = np.sqrt(dy2 + dx**2)
            if (d <= np.abs(rad) and nfit < ndat):
                datx[nfit] = np.array([h,k])
                daty[nfit] = image[ik,ih]
                nfit += 1
    if (nfit > 0):
        return [nfit, datx[0:nfit], daty[0:nfit]]
    return [0, [], []]
# %%
def fit_local_gauss_2d(image, pos, rad, wrap=True):
    '''
    Fits a gaussian peak to a circular local image area.

    Parameters:
        image : numpy.ndarray
            image data
        pos : array (x,y)
            center position for the fit region of interest
        rad : float
            radius of the fit region of interest
            must be larger than 2
        wrap : bool
            flag for using periodic wrap around when crossing
            input image bounds

    Return:
        [best fitting parameters, parameter error estimates]
        fit parameters: [x0, y0, a, bxx, bxy, byy, c], 
        see emilys.functions.peaks.gauss_2d
    '''
    nprm = 7 # number of peak parameters
    ndim = np.array(image.shape)
    assert ndim.size == 2, 'expecting a 2d array as parameter 1'
    assert np.size(pos) == 2, 'expecting a tuple as parameter 2'
    assert np.abs(rad) > 2, 'expecting rad > 2 (parameter 3)'
    (nfit, xfit, yfit) = get_data_in_circroi(image, pos, rad, wrap)
    assert nfit > 2 * nprm, 'insufficient number of image points in region of interest'
    prm0 = [ pos[0], pos[1], np.max(yfit) - np.min(yfit), 
             1./np.abs(rad), 0., 1./np.abs(rad), np.min(yfit)] # initial parameter set
    solprm, solcov = curve_fit(gauss_2d, xfit.T, yfit, prm0) # call scipy.optimize.curve_fit
    solerr = np.sqrt(np.diag(solcov))
    return [solprm, solerr]