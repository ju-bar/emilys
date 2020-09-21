# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:14:00 2020
@author: ju-bar

Linear search function.

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
# %%
from numba import jit # include compilation support
import numpy as np
# %%
def linsrch_minloc1(data, subpix = True):
    '''

    Searches for the for the first local minimum of the input data and returns the index
    of the index where that occurs. A 2nd-order sub-pixel estimate is attempted if
    subpix == True and the initial search didn't stop at either end of the data row.

    Parameters:
        data : numpy.array
            input data, rows of data will be searched, 2d input is supported
        subpix : bool
            switches sub-pixel estimates

    Returns: float or 1d numpy.array depending on dimension of input data

    '''
    ndim = data.shape
    assert (len(ndim)>0 and len(ndim)<3), 'this only supports 1d or 2d input data'
    nrows = 1
    if len(ndim)==2:
        b2d = True
        nrows = ndim[0]
        ndat = ndim[1]
    else:
        b2d = False
        ndat = ndim[0]
    dout = np.zeros(nrows, dtype=float)
    for j in range(0, nrows):
        if (b2d): # set drow from 2d input
            drow = data[j]
        else: # set drow from 1d input
            drow = data
        # find 1st local minimum in drow
        im = 0
        vmin = drow[0]
        for i in range(1, ndat): # loop row data
            if (drow[i] < vmin): # new min
                vmin = drow[i]
                im = i
            else:
                break # data increases, stop
        dout[j] = 1.0 * im # store min loc
        if (subpix and im > 0 and im < ndat-1): # try to estimate min-loc sub-sample
            # determine minimum location of the parabole crossing points (im-1, im, im+1)
            v1 = drow[im-1]
            v2 = drow[im]
            v3 = drow[im+1]
            dout[j] = (4. * im**2 + v1 - v3 +  2. * im * (v1 - 2. * v2 + v3))/(2. * (v1 - 2. * v2 + im * (2. + v1 - v3) + v3))
    #
    if (nrows == 1): return dout[0] # return scalar
    return dout # return vector
