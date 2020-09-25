# -*- coding: utf-8 -*-
"""
Created on Sun Jul 1 15:42:00 2019
@author: ju-bar

Image data access tools

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
# %%
from numba import jit # include compilation support
import numpy as np # include numeric functions
# %%
def image_load(str_file_name, nx, ny, datatype):
    '''
    Loads data from a file and reshapes to an image of ny rows and row length nx.
    '''
    img0 = np.fromfile(str_file_name, dtype=datatype).reshape((ny,nx))
    return img0
# %%
def image_diffractogram(image):
    '''
    Returns the diffractogram I(q)**2 of an image
    '''
    ndim = image.shape
    assert len(ndim)==2, 'this is for 2d images only'
    ndim2 = np.array([ndim[0]>>1,ndim[1]>>1]) # get nyquist index
    imgft = np.fft.fft2(image) / (ndim[0] * ndim[1]) # fourier transform
    ftpowsca = ndim[0] * ndim[1]
    dif0 = (imgft.real**2 + imgft.imag**2) * ftpowsca # absolute square
    dif1 = np.roll(dif0, shift = ndim2, axis = (0,1)) # put dc on [ny/2,nx/2]
    return dif1
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
    # periodic
    il = np.mod(il,dimg[1]) # x
    ih = np.mod(ih,dimg[1])
    jl = np.mod(jl,dimg[0]) # y
    jh = np.mod(jh,dimg[0])
    # sum to bi-linear interpolation
    v = image[jl,il] * (1-fj) * (1-fi) + image[jl,ih] * (1-fj) * fi \
        + image[jh,il] * fj * (1-fi) + image[jh,ih] * fj * fi
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
def image_resample(image, nout, p0in=np.array([0.,0.]), 
                   p0out=np.array([0.,0.]), 
                   sampling=np.array([[1.,0.],[0.,1.]]), ipol=1):
    '''

    Resamples an image on a grid of new dimensions with given shift and sampling

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
            sampling matrix transforming a position in the output to
            a position in the input
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

def maxpos(array):
    '''
    Returns the index of the first array position with maximum value

    Parameters:
        array : numpy.ndarray int, float
            list of values

    Return:
        numpy.array (2,)
            (x,y) location of global maximum
    '''
    nd = array.shape
    assert len(nd)==2, 'this works only for 2d arrays'
    imax = np.argmax(array)
    i = imax%nd[1]
    j = int((imax-i)/nd[1])
    return np.array([i,j])

def com(array):
    '''
    Returns the center of mass of image data

    Parameters:
        array : numpy.ndarray int, float
            list of values

    Return:
        numpy.array (2,)
            (x,y) center of mass
    '''
    nd = array.shape
    assert len(nd)==2, 'this works only for 2d arrays'
    s0 = 0.
    sx = 0.
    sy = 0.
    for j in range(0, nd[0]):
        for i in range(0, nd[1]):
            v = array[j,i]
            s0 += v
            sx += v * i
            sy += v * j
    return np.array([sx/s0,sy/s0])

# %%
@jit
def data_convolute_2d(sfile_fmt, ix, iy, ndata, krn, thr_krn):
    dkrn = krn.shape
    my = dkrn[0]
    mx = dkrn[1]
    apat = np.zeros(ndata).astype(np.float32) # init zero accumulation buffer
    for jy in range(0, my):
        ly = (jy - iy) % my # periodic wrap of the the shifted y index of the kernel
        for jx in range(0, mx):
            lx = (jx - ix) % mx # periodic wrap of the the shifted x index of the kernel
            if (krn[ly,lx] > thr_krn): # only add effective pixels
                sfile_in = sfile_fmt.format(jx,jy)
                bpat = np.fromfile(sfile_in, dtype=np.float32)
                apat += bpat * krn[ly,lx]
    return apat