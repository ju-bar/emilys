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
        fit parameters: [x0, y0, a, rbxx, rbxy, rbyy, c], 
        see emilys.functions.peaks.gauss_2d
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
    datx = np.array(, dtype=float)
    for k in range(ir0[1], ir1[1]+1):
        for h in range(ir0[0], ir1[1]+1):
            

