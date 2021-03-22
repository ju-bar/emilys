# -*- coding: utf-8 -*-
"""
Created on Sun Jul 1 15:42:00 2019
@author: ju-bar

Image data access tools

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
#
from numba import jit # include compilation support
import numpy as np # include numeric functions
#
def image_load(str_file_name, nx, ny, datatype):
    '''
    Loads data from a file and reshapes to an image of ny rows and row length nx.
    '''
    img0 = np.fromfile(str_file_name, dtype=datatype).reshape((ny,nx))
    return img0
#
def image_diffractogram(image):
    '''
    Returns the diffractogram I(q)**2 of an image
    '''
    ndim = image.shape
    assert len(ndim)==2, 'this is for 2d images only'
    ndim2 = np.array([ndim[0]>>1,ndim[1]>>1]) # get nyquist index
    imgft = np.fft.fft2(image) # fourier transform
    ftpowsca = 1. / (ndim[0] * ndim[1])
    dif0 = (imgft.real**2 + imgft.imag**2) * ftpowsca # absolute square
    dif1 = np.roll(dif0, shift = ndim2, axis = (0,1)) # put dc on [ny/2,nx/2]
    return dif1
#
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
#
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
#
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
#
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
#
def image_resample_ft(image, dim_out):
    '''

    Resamples a 2D array image to new dimensions dim_out using Fourier
    interpolation. The main application is to increase the sampling,
    decreasing the sampling is possible, but neglects aliasing.

    Parameters
    ----------
        image : numpy.ndarray
            2-dimensional input data
        dim_out : list or array of 2 ints
            target sampling (number of rows, length of rows)

    Returns
    -------
        numpy.ndarray
            a resampled version of image

    '''
    dim_in = list(image.shape)
    nd = len(dim_in)
    adtype = image.dtype
    assert (nd == 2 and len(dim_out) == 2), "This is meant for 2-dimensional input and output arrays."
    assert (dim_in[nd-1] > 1 and dim_in[nd-2] > 1 and dim_out[0] > 0 and dim_out[1] > 0), "This doesn't work with zero dimensions."
    nix = dim_in[1]
    niy = dim_in[0]
    nix2 = nix >> 1
    niy2 = niy >> 1
    nox = dim_out[1]
    noy = dim_out[0]
    scale = nox * noy / nix / niy
    nox2 = nox >> 1
    noy2 = noy >> 1
    ntx2 = min(nix2, nox2) # transfer nyquist row length
    nty2 = min(niy2, noy2) # transfer nyquist column length
    img_out = np.full(list(dim_out), 0., dtype = adtype)
    img_in_ft = np.fft.fft2(image)
    img_out_ft = np.full(list(dim_out), 0. + 0.j, dtype=img_in_ft.dtype)
    img_out_ft[0:nty2,0:ntx2] = img_in_ft[0:nty2,0:ntx2]
    img_out_ft[0:nty2,nox-ntx2:nox] = img_in_ft[0:nty2,nix-ntx2:nix]
    img_out_ft[noy-nty2:noy,0:ntx2] = img_in_ft[niy-nty2:niy,0:ntx2]
    img_out_ft[noy-nty2:noy,nox-ntx2:nox] = img_in_ft[niy-nty2:niy,nix-ntx2:nix]
    if np.iscomplex([image[0,0]])[0]:
        img_out[:,:] = np.fft.ifft2(img_out_ft).astype(adtype) * scale
    else:
        img_out[:,:] = (np.fft.ifft2(img_out_ft).real).astype(adtype) * scale
    return img_out
#
def image_ser_resample_ft(img_ser, dim_out):
    '''

    Resamples a series of 2D array images to new dimensions dim_out
    using Fourier interpolation. The main application is to increase
    the sampling, decreasing the sampling is possible, but neglects
    aliasing.

    Parameters
    ----------
        image : numpy.ndarray
            3-dimensional input data
        dim_out : list or array of 2 ints
            target sampling (number of rows, length of rows)

    Returns
    -------
        numpy.ndarray
            a resampled version of image

    '''
    dim_in = list(img_ser.shape)
    nd = len(dim_in)
    adtype = img_ser.dtype
    assert (nd == 3 and len(dim_out) == 2), "This is meant for a series of 2-dimensional input and output arrays."
    assert (dim_in[nd-1] > 1 and dim_in[nd-2] > 1 and dim_out[0] > 0 and dim_out[1] > 0), "This doesn't work with zero dimensions."
    dim_ou = list(dim_in)
    dim_ou[nd-2] = dim_out[0]
    dim_ou[nd-1] = dim_out[1]
    nimg = dim_in[0]
    img_ser_out = np.full(dim_ou, 0., dtype = adtype)
    for i in range(0, nimg):
        img_ser_out[i,:,:] = image_resample_ft(img_ser[i], dim_out)
    return img_ser_out
#
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
#
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
#
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

#
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
#
def convolute_2d(data, kernel):
    """

    Convolutes the 2d array data by the 2d kernel.

    Parameters
    ----------
        data : numpy.ndarray
            2-dimensional input data
        kernel : numpy.ndarray
            2-dimensional input kernel

    """
    nd = data.shape
    nk = kernel.shape
    assert len(nd)==2, 'input data must have two dimensions'
    assert len(nk)==2, 'input kernel must have two dimensions'
    assert (nd[0]==nk[0] and nd[1]==nk[1]), 'input data and kernel must be of identical size'
    data_ft = np.fft.fft2(data)
    kernel_ft = np.fft.fft2(kernel)
    return np.fft.ifft2(data_ft * kernel_ft.conjugate())
#
def convolute_img_ser(img_ser, kernel):
    nd = img_ser.shape
    nk = kernel.shape
    assert (len(nd)==2 or len(nd)==3), 'input data must have two or three dimensions'
    assert len(nk)==2, 'input kernel must have two dimensions'
    if (len(nd)==2): return convolute_2d(img_ser, kernel).real
    assert (nd[1]==nk[0] and nd[2]==nk[1]), 'input images and kernel must be of identical size'
    kernel_ft = np.fft.fft2(kernel)
    img_out = img_ser.copy()
    for iimg in range(0, nd[0]):
        data_ft = np.fft.fft2(img_ser[iimg])
        img_out[iimg,:,:] = np.fft.ifft2(data_ft * kernel_ft.conjugate()).real # warning this assumes real valued image input
    return img_out