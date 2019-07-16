# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 22:13:00 2019
@author: ju-bar

Image convolution kernel routines

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
from numba import jit # include compilation support
import numpy as np
# %%
@jit # compilation decorator, do this, if you intend to call the function several timed
def bwl_gaussian(shape=None, size=0.):
    '''
    Calculates a bandwidth limiting kernel of Gausian shape with
    given size in pixels. The kernel is returned in the original
    space of the image and is normalized to a total of 1.

    Parameters:
        shape : array (2,)
            shape of the image, number of pixels (rows, columns)
        size : float or array of floats
            size of the bandwidth limit in the original space of the
            image.
            1 float: round gaussian of given width width
                (a = c = size, b = 0)
            2 floats: elliptical gaussian with main axis along the
                    image rows and columns and respective rms widths
                (a = size[0], c = size[1], b = 0)
            3 or more floats: elliptical gaussian with
                (a = size[0], b = size[1], c = size[2])
            
    Return:
        numpy.ndarray, shape=shape, dtype=float
            2D normalized kernel array of requested shape

    Notes:
        Centered bivariate normal distribution,
        see emilys.functions.distributions.bivnorm
        Uses the same function, but simplified code

        p(x,y;a,b,c) = Exp[ -1/2 * (c^2 x^2 + 2 b x y + a^2 y^2) / (a^2 c^2 - b^2)] 
            / (2 Pi sqrt(a^2 c^2 - b^2))

        constraints: abs(a) > 0 && abs(c) > 0 && a^2 c^2 - b^2 != 0
    '''
    assert np.size(shape)==2, 'expecting parameter 1 as tuple, shape'
    nd = np.array(shape).astype(int)
    assert nd[0] > 0, 'expecting number of rows in first element of parameter 1, shape'
    assert nd[1] > 0, 'expecting number of columns in first element of parameter 1, shape'
    ls = np.array([0.,0.,0.]) # a, b, c
    if (np.size(size)==1):
        ls[0] = np.abs(size)
        ls[2] = np.abs(size)
        assert ls[0] > 0, 'expecting finite size in parameter 2, size'
    elif (np.size(size)==2):
        ls[0] = np.abs(size[0])
        ls[2] = np.abs(size[1])
        assert ls[0] > 0 and ls[2] > 0, 'expecting finite size in parameter 2, size'
    elif (np.size(size)==3):
        ls[0] = np.abs(size[0])
        ls[1] = size[1]
        ls[2] = np.abs(size[2])
        assert ls[0] > 0 and ls[2] > 0, 'expecting finite size in parameter 2, size'
        assert ls[0]**2 * ls[2] ** 2 - ls[1]**2 > 0, \
            '3 size parameters do not define a peak: a^2 c^2 - b^2 <= 0'
    else:
        assert False, 'expecting finite size in parameter 2, size'
    d1 = 1. / (ls[0]**2 * ls[2] ** 2 - ls[1]**2)
    d2 = 0.5 / np.sqrt(d1) / np.pi
    d1 = -0.5 * d1
    kernel = np.zeros(nd, dtype=float)
    nd2 = np.array([int((nd[0]-nd[0]%2)/2), int((nd[1]-nd[1]%2)/2)], dtype=int)
    xi = np.zeros(nd[1], dtype=int)
    for i in range(0, nd[1]):
        xi[i] = (i + nd2[1])%nd[1] - nd2[1]
    norm = 0.
    for j in range(0, nd[0]):
        yj = (j + nd2[0])%nd[0] - nd2[0]
        argy = yj**2 * ls[0]**2
        for i in range(0, nd[1]):
            arg = xi[i]**2 * ls[2]**2 + argy + 2 * xi[i] * yj * ls[1]
            v = d2 * np.exp( d1 * arg )
            norm = norm + v
            kernel[j,i] = v
    kernel = kernel / norm
    return kernel
