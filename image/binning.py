# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:23:00 2020
@author: ju-bar

Binning functions

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
#
import numpy as np
#
def hash_bin(x, n, x_hash):
    """

    Accumulates data form x in n bins according to the access directive x_hash,
    accumulating: y[x_hash[i]] += x[i].

    Parameters
    ----------
        x : numpy array, float, shape = (..., m,)
            input data
        n : int
            number of output bins
        x_hash : numpy array, int, shape = (m,)
            access directive

    Returns
    -------
        numpy array, float, shape = (..., n,)

    """
    nx = x.shape
    m = nx[len(nx)-1]
    assert m == len(x_hash), 'axis 0 of input x and x_hash must be of the same length'
    y = np.zeros(nx)
    for i in range(len(x_hash)): # for each data, ... this may do random access on y and slow down
        y[...,x_hash[i]] += x[...,i]
    return y