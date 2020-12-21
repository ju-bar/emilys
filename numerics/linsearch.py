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
import numpy as np
# %%
def linsrch_minloc1(data, subpix = True):
    '''

    Searches for the first local minimum of the input data and returns the index
    where that occurs. A 2nd-order sub-pixel estimate is attempted if
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

# %%
def maxloc_com(y, yerr=0., x0=-1., xrng = 10.):
    calc_err = False
    nd = y.shape
    assert (len(nd)<=2 and len(nd)>=1), 'expecting 1d or 2d array is input data'
    if (np.size(yerr)==np.size(y)):
        nds = yerr.shape
        assert len(nds)==len(nd), 'input y and yerr must be of same dimension'
        assert nds[0]==nd[0], 'input y and yerr must be of same shape'
        if (len(nd)==2):
            assert nds[1]==nd[1], 'input y and yerr must be of same shape'
        calc_err = True
    nrow = 1
    if (len(nd) == 2):
        nrow = nd[0]
        ndim = nd[1]
    else:
        ndim = nd[0]
    ly = y.reshape((nrow,ndim))
    lys = yerr.reshape((nrow,ndim))
    if (np.size(x0) == nrow):
        lx0 = x0.copy()
    else:
        lx0 = np.argmax(ly,axis=1)
    # ---- #
    lc = np.zeros((nrow,2)).astype(np.float64)
    lc[:,0] = lx0
    lit = np.zeros(nrow).astype(int)
    lrep = np.zeros(nrow).astype(int) + 1
    nrep = np.sum(lrep) # get number of rows to keep on iterating
    while (nrep > 0):
        lcprev = lc.copy() # previous result
        for irow in range(0, nrow):
            if (lrep[irow] == 0): continue # skip row, it is not marked for further iteration
            lit[irow] += 1
            lxi = np.arange(0,ndim).astype(np.float64)
            lwi = np.exp( - (lxi-lc[irow,0])**2 / (2. * xrng**2))
            lyi = ly[irow]
            m0 = np.sum(lwi*lyi)
            m1 = np.sum(lwi*lyi*lxi)
            lc[irow,0] = m1 / m0
            if (calc_err): # error propagation
                lsi = lys[irow]
                s1 = np.sum((lsi*lwi*lxi)**2) 
                s2 = np.sum((lsi*lwi)**2) * m1**2 / m0**2
                s3 = -2. * np.sum((lsi*lwi)**2 * lxi) * m1 / m0
                lc[irow,1] = (s1 + s2 + s3) / m0**2
            if ((np.abs(lc[irow,0] - lcprev[irow,0]) < 0.001) or lit[irow] > 100): # convergence check
                lrep[irow] = 0 # set halt to iteration for this row
        # ----- loop on rows
        nrep = np.sum(lrep) # get number of rows to keep on iterating
    # --- iterator for convergence
    print(lit)
    return lc