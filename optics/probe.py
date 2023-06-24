# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 16:15:00 2023
@author: ju-bar

Implementations of calculating and manipulating
electron probe functions

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

import numpy as np
from numba import njit
from emilys.optics.aperture import aperture

@njit
def get_probe_q(a, dq, q0, qa):
    ndim = np.array(a.shape, dtype=np.int32)
    ndim2 = ndim >> 1
    lq0 = np.array([q0[1], q0[0]], dtype=np.float64)
    aq1 = dq[1] * (((np.arange(0, ndim[1]) + ndim2[1]) % ndim[1]) - ndim2[1])
    aq2 = dq[0] * (((np.arange(0, ndim[0]) + ndim2[0]) % ndim[0]) - ndim2[0])
    s = 0.0
    smt = np.float64(0.5 * (dq[0]+dq[1]))
    for i in range(0, ndim[0]):
        for j in range(0, ndim[1]):
            vq = np.array([aq1[j],aq2[i]], dtype=np.float64)
            v = aperture(vq, lq0, np.float64(qa), smt)
            a[i,j] = v
            s += v*v
    a[:,:] = a / np.sqrt(s)

def get_probe(a, dq, q0, qa):
    ndim = np.array(a.shape, dtype=np.int32)
    b = np.zeros(ndim, dtype=np.complex128)
    get_probe_q(b, dq, q0, qa)
    a[:,:] = np.fft.ifft2(b) * np.sqrt(ndim[0] * ndim[1])
    del(b)

@njit
def shift_probe_q(a_q_shift, a_q, dq, dr):
    ndim = np.array(a_q.shape, dtype=np.int32)
    ndim2 = ndim >> 1
    pq1 = -2.0J * np.pi * dr[1] * dq[1] * (((np.arange(0, ndim[1]) + ndim2[1]) % ndim[1]) - ndim2[1])
    pq2 = -2.0J * np.pi * dr[0] * dq[0] * (((np.arange(0, ndim[0]) + ndim2[0]) % ndim[0]) - ndim2[0])
    for i in range(0, ndim[0]):
        for j in range(0, ndim[1]):
            a_q_shift[i,j] = a_q[i,j] * np.exp(pq2[i] + pq1[j])
