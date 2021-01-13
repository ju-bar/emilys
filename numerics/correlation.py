# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:04:00 2021
@author: ju-bar

Correlation utilities

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np

def corr_norm(a, b):
    """
    Calculates the normalized correlation coefficient of two arrays.

    Parameters
    ----------
        a : numpy array
            input array
        b : numpy array
            input array to be correlated with a

    Returns
    -------
        float
            correlation coefficient [-1., 1.]
    """
    fa = a.flatten().astype(np.dtype('f8'))
    fb = b.flatten().astype(np.dtype('f8'))
    assert len(fa) == len(fb), 'This is for equal size arrays only.'
    ma = np.mean(fa)
    fa_0 = fa - ma
    ssa = np.sum(fa_0**2)
    mb = np.mean(fb)
    fb_0 = fb - mb
    ssb = np.sum(fb_0**2)
    if ssb * ssa == 0.: return 0.
    return np.dot(fa_0/np.sqrt(ssa), fb_0/np.sqrt(ssb))