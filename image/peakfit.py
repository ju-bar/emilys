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
import emilys.functions.peaks as pks
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
    ndat = (ir1[0] - ir0[0] + 1) * (ir1[1] - ir0[1] + 1)
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
        for h in range(ir0[0], ir1[0]+1):
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
                nfit = nfit + 1
    if (nfit > 0):
        return [nfit, datx[0:nfit], daty[0:nfit]]
    return [0, [], []]
# %%
def get_values_in_circroi(image, pos, rad, wrap=True):
    '''
    Extracts values in a circular region of interest from an image.

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
        [number of image points n, image values]
        [int, numpy.ndarray (n)]
    '''
    ndim = np.array(image.shape)
    assert ndim.size == 2, 'expecting a 2d array as parameter 1'
    assert np.size(pos) == 2, 'expecting a tuple as parameter 2'
    assert np.abs(rad) > 2, 'expecting rad > 2 (parameter 3)'
    icent = np.array(pos).astype(int)
    irad = np.ceil(np.abs(rad)) + 1
    ir0 = icent - np.array([irad,irad], dtype=int)
    ir1 = icent + np.array([irad,irad], dtype=int)
    ndat = (ir1[0] - ir0[0] + 1) * (ir1[1] - ir0[1] + 1)
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
        for h in range(ir0[0], ir1[0]+1):
            ih = h
            if wrap:
                ih = int( h%ndim[1] )
            else:
                if (h < 0 or h >= ndim[1]):
                    continue # skip this columns
            dx = h - pos[0]
            d = np.sqrt(dy2 + dx**2)
            if (d <= np.abs(rad) and nfit < ndat):
                daty[nfit] = image[ik,ih]
                nfit = nfit + 1
    if (nfit > 0):
        return [nfit, daty[0:nfit]]
    return [0, []]
# %%
def fit_local_gauss_2d(image, pos, rad, wrap=True, imagesigma=None, method='lm', debug=False):
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
        imagesigma : numpy.ndarray or None
            image data error estimates
        method : str
            optimization methods, see scipy.optimize.curve_fit
        debug : bool
            flag text output

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
    assert nfit > 2 * nprm, 'insufficient number of image points '\
        f'({nfit} of {2*nprm}) in region of interest'
    if debug:
        print(f'dbg (fit_local_gauss_2d): nfit = {nfit}')
    ysigu = None
    if imagesigma is not None: # also extract image sigma values
        (nfit2, ysig) = get_values_in_circroi(imagesigma, pos, rad, wrap)
        ysigu = np.array(ysig).astype(np.float64)
        assert nfit == nfit2, 'internal data size conflict with sigma data'
    # initial parameter set
    prm0 = [ pos[0], pos[1], np.amax(yfit) - np.amin(yfit), 
             1.2/np.abs(rad), 0.0001/np.abs(rad), 1.2/np.abs(rad), np.amin(yfit)]
    # bounds
    bds  = [[pos[0]-0.7*rad, pos[1]-0.7*rad, 0.1*prm0[2], 0.1*prm0[3], -10/np.abs(rad), 0.1*prm0[5], -np.inf],
            [pos[0]+0.7*rad, pos[1]+0.7*rad, 10*prm0[2], 10*prm0[3], 10/np.abs(rad), 10*prm0[5], np.inf]]
    if debug:
        print('dbg (fit_local_gauss_2d): prm0 =', prm0)
    sol = curve_fit(pks.gauss_2d, 
                    np.array(xfit.T).astype(np.float64), 
                    np.array(yfit).astype(np.float64), 
                    np.array(prm0).astype(np.float64),
                    jac=pks.gauss_2d_jac,
                    sigma=ysigu, 
                    method='trf', 
                    bounds=bds
                    ) # call scipy.optimize.curve_fit
    if debug:
        print('dbg (fit_local_gauss_2d): prm =', sol[0])
    solerr = np.sqrt(np.diag(sol[1]))
    if debug:
        print('dbg (fit_local_gauss_2d): std =', solerr)
    return [sol[0], solerr]
# %%
def com_local(image, pos, rad, wrap=True, debug=False):
    '''
    Measures the center of mass in a circular local image area.

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
        [center of mass x, y]
        
    '''
    ndim = np.array(image.shape)
    assert ndim.size == 2, 'expecting a 2d array as parameter 1'
    assert np.size(pos) == 2, 'expecting a tuple as parameter 2'
    assert np.abs(rad) > 2, 'expecting rad > 2 (parameter 3)'
    (nfit, xfit, yfit) = get_data_in_circroi(image, pos, rad, wrap)
    assert nfit > 9, 'insufficient number of image points '\
        f'({nfit}) in region of interest'
    if debug:
        print(f'dbg (com_local): nfit = {nfit}')
    com = pos # initialize result
    if debug:
        print('dbg (com_local): com0 =', com)
    m0 = np.sum(yfit) # total mass
    m1 = np.dot(xfit.T, yfit)
    com = m1 / m0 # first moment
    if debug:
        print('dbg (com_local): com =', com)
    return com