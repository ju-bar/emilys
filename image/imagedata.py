# -*- coding: utf-8 -*-
"""
Created on Sun Jul 1 15:42:00 2019
@author: ju-bar
"""
# %%
from numba import jit # include compilation support
import numpy as np # include numeric functions
# %%
def image_at_nn(image, pos):
    '''

    Returns nearest-neighbor interpolation of image at position pos.

    '''
    dimg = image.shape
    ic = np.floor(pos[::-1]).astype(int) # reverse sequence of coordinates
    j = np.mod(ic[0],dimg[0])
    i = np.mod(ic[1],dimg[1])
    v = image[j,i]
    return v
# %%
@jit
def image_at_bilin(image, pos):
    '''

    Returns a bi-linear interpolation of image at position pos.

    '''
    dimg = image.shape
    il = int(np.floor(pos[0])) # x
    ih = il + 1
    fi = pos[0] - il
    jl = int(np.floor(pos[1])) # y
    jh = jl + 1
    fj = pos[1] - jl
    # peridic
    il = np.mod(il,dimg[1]) # x
    ih = np.mod(ih,dimg[1])
    jl = np.mod(jl,dimg[0]) # y
    jh = np.mod(jh,dimg[0])
    # sum to bi-linear interpolation
    v = image[jl,il] * (1-fj) * (1-fi) + image[jl,ih] * (1-fj) * fi + image[jh,il] * fj * (1-fi) + image[jh,ih] * fj * fi
    return v
# %%
def image_at(image, pos, ipol=1):
    '''

    Returns the image intensity at a given image position.

    Parameters:
        image : numpy.array of 2 dimensions
            data on a regular grid
        pos : numpy.array of 1 dimension
            2-dimensional coordinate (x,y)
        ipol : int
            interpolation order, default = 1
            0 = nearest neighbor
            1 = bi-linear

    Remark:
        Doesn't check for reasonable input ranges.
        Applies periodic boundary conditions

    '''
    ipol_switcher = {
        0 : image_at_nn(image, pos),
        1 : image_at_bilin(image, pos)
    }
    return ipol_switcher.get(ipol, image_at_bilin(image, pos))
# %%
@jit
def image_resample(image, nout, p0in=np.array([0.,0.]), p0out=np.array([0.,0.]), sampling=np.array([[1.,0.],[0.,1.]]), ipol=1):
    '''

    Resamples and image on a grid of new dimensions with given shift and sampling

    Parameters:
        image : numpy.array of 2 dimensions
            data on a regular grid
        nout : array (size=2)
            new dimensions (ny,nx)
        p0in : numpy.array, size=2
            reference position in input array (x,y)
        p0out : numpy.array, size=2
            reference position in output array (x,y)
        sampling : numpy.array, shape(2,2)
            sampling matrix transforming a position in the output to a position in the input
        ipol : integer
            interpolation order

    '''
    image_out = np.zeros((nout[0],nout[1]))
    for j in range(0, nout[0]):
        for i in range(0, nout[1]):
            pout = np.array([i,j]) - p0out
            pin = np.dot(sampling,pout) - p0in
            image_out[j,i] = image_at(image, pin, ipol)
    return image_out
# %%
@jit
def image_pos_sum(image, lpoints, ipol=1):
    '''

    Calculates the sum of image intensities at points given by lpoints.

    Parameters:
        image : numpy.array of 2 dimensions
            data on a regular grid
        lpoints : numpy.array of 2 dimensions
            list of 2-dimensional coordinates defining points in image
        ipol : integer
            interpolation order

    Returns:
        Sum of intensities at the given points

    '''
    dpts = lpoints.shape
    npts = dpts[0]
    spt = 0.
    for l in range(0, npts):
        spt += image_at(image, lpoints[l], ipol)
    return spt