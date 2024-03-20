# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 15:08:00 2024
@author: ju-bar

data alignment

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np

def get_row_align(a, ref_row=0):
    """

    Calculates shifts that align rows to each other based on
    row-to-row cross-correlation results.

    Parameters
    ----------
        a : numpy.array, shape(n,m)
            a 2d data set, with n rows of length m
        ref_row : int
            reference row, that will not be shifted
    
    Returns
    -------
        numpy.array, shape(n), dtype=int
            shift values for each row in pixels

    """
    ndim = np.array(a.shape)
    assert len(ndim)==2, "expecting 2d input data"
    n = ndim[0]
    m = ndim[1]
    assert m > 2, "expecting second dimension > 2"
    b = np.zeros(n, dtype=int)
    if n > 1:
        lsh = (np.fft.fftfreq(m) * m).astype(int)
        amean = np.mean(a, axis=1)
        asdev = np.std(a, axis=1)
        assert np.amin(asdev) > 0.0, 'no contrast in some rows'
        xr = (a[ref_row] - amean[ref_row]) / asdev[ref_row]
        ftr = np.fft.fft(xr)
        for i in range(0, n-1):
            if i == ref_row: continue
            xi = (a[i] - amean[i]) / asdev[i]
            fti = np.fft.fft(xi)
            xcor = np.real(np.fft.ifft(fti * np.conjugate(ftr)))
            jmax = np.argmax(xcor)
            print(i, jmax)
            b[i] = lsh[jmax]
    return b

def row_align(a, row_shifts):
    b = a.copy()
    ndim = np.array(a.shape)
    n = len(row_shifts)
    assert ndim[0] == n, "expecting number of rows in a to be equal to len(row_shifts)"
    for i in range(0, n):
        b[i] = np.roll(a[i], row_shifts[i], axis=0)
    return b