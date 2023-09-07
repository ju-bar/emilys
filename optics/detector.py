# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 12:04:00 2019
@author: ju-bar

Calculation of 2-dimensional detector functions

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

import numpy as np
from numba import njit, int64, float64
from emilys.optics.aperture import aperture

@njit(float64[:](int64[:],float64[:],float64[:],float64,float64,float64,float64))
def annular_detector(dim, dq, q0, q_min, q_max, smooth, threshold):
    '''

    annular_detector

    Calculates an annular detector function using FFT
    organization of spatial frequencies.

    Parameters
    ----------
        TBA
    
    Returns
    -------
        TBA
    
    '''
    det = np.zeros(dim, dtype=np.float64)
    dim2 = np.arra(dim, dtype=np.int64) >> 1
    qy = np.zeros(dim[0], dtype=np.float64)
    for i in range(0, dim[0]): # qy
        qy[i] = dq[0] * ((i + dim2[0]) % dim[0] - dim2[0])
    qx = np.zeros(dim[1], dtype=np.float64)
    for i in range(0, dim[1]): # qx
        qx[i] = dq[1] * ((i + dim2[1]) % dim[1] - dim2[1])
    psmt = smooth * 0.5 * (dq[0] + dq[1]) # use average smoothness in the hope of not too much anisotropy
    for i in range(0, dim[0]): # qy
        for j in range(0, dim[1]): # qx
            vq = np.array([qx[j],qy[i]], dtype=np.float64)
            vi = 0.0
            if q_min > 0:
                vi = aperture(vq, q0, q_min, psmt)
            va = aperture(vq, q0, q_max, psmt)
            v = va - vi

            if v > threshold:
                det[i,j] = v
    return det