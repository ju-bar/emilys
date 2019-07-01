# -*- coding: utf-8 -*-
"""
Created on Sun Jul 1 15:42:00 2019
@author: ju-bar
"""
# %%
from numba import jit # include compilation support
import numpy as np # include numeric functions
# %%
def image_at(image, pos):
    '''

    Returns the image intensity at a given image position.

    Parameters:
        image : np.array of 2 dimensions
            data on a regular grid
        pos : np.array of 1 dimension
            2-dimensional coordinate

    Remark:
        Doesn't check for reasonable input ranges.
        Applies periodic boundary conditions

    '''
    dimg = image.shape
    j = np.mod(int(np.around(pos[0])),dimg[0])
    i = np.mod(int(np.around(pos[1])),dimg[1])
    v = image[j,i]
    return v
# %%
@jit
def image_pos_sum(image, lpoints):
    '''

    Calculates the sum of image intensities at points given by lpoints.

    Parameters:
        image : np.array of 2 dimensions
            data on a regular grid
        lpoints : np.array of 2 dimensions
            list of 2-dimensional coordinates defining points in image

    Returns:
        Sum of intensities at the given points

    '''
    dpts = lpoints.shape
    npts = dpts[0]
    spt = 0.
    for l in range(0, npts):
        spt += image_at(image, lpoints[l])
    return spt