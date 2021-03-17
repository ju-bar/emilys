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
import emilys.optics.aperture as apert
# %%
@jit # compilation decorator, do this, if you intend to call the function several timed
def bwl_gaussian(shape=None, size=0.):
    '''
    Calculates a bandwidth limiting kernel of Gaussian shape with
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
        
        The returned distribution is centered at the [0,0] item of the array.
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
# %%
def source_delta(kernel):
    '''
    Calculates delta function distribution in a given array.

    Parameters:
        kernel: numpy.ndarray dtype=float
            array receiving distribution values
    
    Return:
        modifies the array kernel
    '''
    kernel[0,0] = 1.
    return 0
# %%
@jit
def source_normal(kernel, samp, src_size):
    '''
    Calculates normal distribution values in a given array.

    Parameters:
        kernel: numpy.ndarray dtype=float
            array receiving distribution values
        samp: numpy.ndarray shape=(2,) dtype=float
            row and column sampling rates
        src_size: float
            source size (HWHM) in physical units
    
    Return:
        modifies the array kernel
    '''
    srcsz = np.abs(src_size)
    nd = np.array(kernel.shape)
    nd2 = [nd[0]>>1,nd[1]>>1]
    krad = 3 * srcsz
    kr2 = krad**2
    nkx2 = int(min(np.ceil(krad / samp[0]),nd2[1]))
    nky2 = int(min(np.ceil(krad / samp[1]),nd2[0]))
    kprm = -np.log(2.) / srcsz**2
    norm = 0.
    for j in range(-nky2, nky2+1):
        j1 = j%nd[0]
        ry = samp[1] * j
        ry2 = ry**2
        for i in range(-nkx2, nkx2+1):
            i1 = i%nd[1]
            rx = samp[0] * i
            r2 = ry2 + rx**2
            if (r2 < kr2):
                kval = np.exp(kprm * r2)
                norm = norm + kval
                kernel[j1,i1] = kval
    kernel[...] = kernel / norm
    return 0
# %%
@jit
def source_cauchy(kernel, samp, src_size):
    '''
    Calculates Cauchy distribution values in a given array.

    Parameters:
        kernel: numpy.ndarray dtype=float
            array receiving distribution values
        samp: numpy.ndarray shape=(2,) dtype=float
            row and column sampling rates
        src_size: float
            source size (HWHM) in physical units
    
    Return:
        modifies the array kernel
    '''
    srcsz = np.abs(src_size)
    nd = np.array(kernel.shape)
    nd2 = ((nd - nd%2)/2).astype(int)
    krad = 10 * srcsz
    kr2 = krad**2
    nkx2 = int(min(np.ceil(krad / samp[0]),nd2[1]))
    nky2 = int(min(np.ceil(krad / samp[1]),nd2[0]))
    rprm = 1.3047660265 * srcsz
    afac = 0.5 * rprm / np.pi
    kprm = rprm**2
    kap = 1.5
    norm = 0.
    for j in range(-nky2, nky2+1):
        j1 = j%nd[0]
        ry = samp[1] * j
        ry2 = ry**2
        for i in range(-nkx2, nkx2+1):
            i1 = i%nd[1]
            rx = samp[0] * i
            r2 = ry2 + rx**2
            if (r2 < kr2):
                kval = afac / (r2 + kprm)**kap
                norm = norm + kval
                kernel[j1,i1] = kval
    kernel[...] = kernel / norm
    return 0
# %%
@jit
def source_disk(kernel, samp, src_size):
    '''
    Calculates disk distribution values in a given array.

    Parameters:
        kernel: numpy.ndarray dtype=float
            array receiving distribution values
        samp: numpy.ndarray shape=(2,) dtype=float
            row and column sampling rates
        src_size: float
            source size in physical units
    
    Return:
        modifies the array kernel
    '''
    srcsz = np.abs(src_size)
    nd = np.array(kernel.shape)
    nd2 = ((nd - nd%2)/2).astype(int)
    krad = 1.5 * srcsz
    nkx2 = int(min(np.ceil(krad / samp[0]),nd2[1]))
    nky2 = int(min(np.ceil(krad / samp[1]),nd2[0]))
    kprm = srcsz
    norm = 0.
    for j in range(-nky2, nky2+1):
        j1 = j%nd[0]
        ry = samp[1] * j
        ry2 = ry**2
        for i in range(-nkx2, nkx2+1):
            i1 = i%nd[1]
            rx = samp[0] * i
            rm = np.sqrt(ry2 + rx**2)
            if (rm < krad):
                kval =  0.5 - 0.5 * np.tanh((rm/kprm - 1.0) * 100.0)
                norm = norm + kval
                kernel[j1,i1] = kval
    kernel[...] = kernel / norm
    return 0
# %%
def source_distribution(shape, samp, src_size, src_type=1):
    '''
    Calculates a normalized source distribution function for convolution
    of incoherent images.

    Parameters:
        shape : array, int, (2,)
            shape of the image, number of pixels (rows, columns)
        samp : array, float (2,)
            sampling rates along image rows and columns (x,y) in
            physical units
        src_size : float
            source size (HWHM) in physical units
        src_type : int
            distribution type
            0 = delta function = size ignored
            1 = normal distribution
            2 = Cauchy (Lorentzian) distribution
            3 = Disk (Sigmoidal) distribution

    Return:
        numpy.ndarray, shape=shape, dtype=float
            2D normalized kernel array of requested shape
    
    Notes:
        This function implements round source distributions only.
        The returned distribution is centered at the [0,0] item of the array.
    '''
    assert np.size(shape)==2, 'expecting parameter 1 as tuple, shape'
    nd = np.array(shape).astype(int)
    assert nd[0] > 0, 'expecting number of rows in first element of parameter 1, shape'
    assert nd[1] > 0, 'expecting number of columns in first element of parameter 1, shape'
    ntuse = int(src_type)
    srcsz = np.abs(src_size)
    if (srcsz == 0.): ntuse = 0
    assert ntuse >= 0 and ntuse <=3, 'expecting integer parameter src_type in range 0 ... 3'
    kernel = np.zeros(nd, dtype=float)
    if (ntuse == 0):
        source_delta(kernel)
    elif (ntuse == 1):
        source_normal(kernel, samp, srcsz)
    elif (ntuse == 2):
        source_cauchy(kernel, samp, srcsz)
    elif (ntuse == 3):
        source_disk(kernel, samp, srcsz)
    return kernel

# %%
@jit
def filter_normal_iso_2d(ndim, samp, flt_size):
    '''
    Returns a Gaussian isotropic filter kernel for a low-pass in an array of given dimension.

    Parameters:
        ndim: numpy.ndarray shape=(2,) dtype=int
            kernel dimensions (nrows, ncolumns), (ny, nx)
        samp: numpy.ndarray shape=(2,) dtype=float
            row and column sampling rates (sx, sy)
        flt_size: float
            filter kernel size (HWHM) in physical units
    
    Return:
        kernel: numpy.ndarray shape=(nrows, ncolumns), dtype=float
    '''
    srcsz = np.abs(flt_size)
    kernel = np.zeros((ndim[0], ndim[1]), dtype=float)
    ndim2 = np.array([ndim[0] >> 1, ndim[1] >> 1])
    krad = 4 * srcsz # calculate filter kernel up to 4 sigma
    kr2 = krad**2
    nkx2 = int(min(np.ceil(krad / samp[0]),ndim2[1]))
    nky2 = int(min(np.ceil(krad / samp[1]),ndim2[0]))
    kprm = -np.log(2.) / srcsz**2
    for j in range(-nky2, nky2+1):
        j1 = j % ndim[0]
        ry = samp[1] * j
        ry2 = ry**2
        for i in range(-nkx2, nkx2+1):
            i1 = i % ndim[1]
            rx = samp[0] * i
            r2 = ry2 + rx**2
            if (r2 < kr2):
                kval = np.exp(kprm * r2)
                kernel[j1,i1] = kval
    return kernel
# %%
@jit
def filter_normal_2d(ndim, samp, flt_size_a, flt_size_b, flt_dir):
    '''
    Returns a Gaussian anisotropic filter kernel for a low-pass in an array of given dimension.
    The filter is returned with value 1 at pixel [0,0]. Normalization should be done numerically.
    The analytical norm if treated as distribution is np.pi * flt_size_a * flt_size_b / np.log(2.).

    Parameters:
        ndim: numpy.ndarray shape=(2,) dtype=int
            kernel dimensions (nrows, ncolumns), (ny, nx)
        samp: numpy.ndarray shape=(2,) dtype=float
            row and column sampling rates (sx, sy)
        flt_size_a: float
            filter kernel size (HWHM) in physical units along orientation a
        flt_size_b: float
            filter kernel size (HWHM) in physical units along orientation b perpendicular to a
        flt_dir: float
            filter kernel a axis orientation to the grid row axis in radians
    
    Return:
        kernel: numpy.ndarray shape=(nrows, ncolumns), dtype=float
    '''
    sz_a = np.abs(flt_size_a)
    sz_b = np.abs(flt_size_b)
    sz_d = flt_dir
    kernel = np.zeros((ndim[0], ndim[1]), dtype=float)
    ndim2 = np.array([ndim[0] >> 1, ndim[1] >> 1])
    krad = 4 * np.amax([sz_a, sz_b]) # calculate filter kernel up to 4 sigma
    kr2 = krad**2
    nkx2 = int(min(np.ceil(krad / samp[0]), ndim2[1]))
    nky2 = int(min(np.ceil(krad / samp[1]), ndim2[0]))
    kprm = -0.5 * np.log(2.) / (sz_a**2 * sz_b**2)
    vprmp = sz_a**2 + sz_b**2
    vprmn = sz_a**2 - sz_b**2
    sdir2 = np.sin(2. * sz_d)
    cdir2 = np.cos(2. * sz_d)
    for j in range(-nky2, nky2+1):
        j1 = j % ndim[0]
        ry = samp[1] * j
        ry2 = ry**2
        for i in range(-nkx2, nkx2+1):
            i1 = i % ndim[1]
            rx = samp[0] * i
            rx2 = rx * rx
            r2 = ry2 + rx2
            if (r2 < kr2):
                kval = np.exp(kprm * (vprmp * r2 - vprmn * ((rx2 - ry2) * cdir2 + 2. * rx * ry * sdir2)))
                kernel[j1,i1] = kval
    return kernel
# %%
@jit
def filter_exp4_2d_iso(ndim, samp, flt_size):
    '''
    Returns an exponential g^4 isotropic filter kernel for a low-pass in an array of given dimension.

    Parameters:
        ndim: numpy.ndarray shape=(2,) dtype=int
            kernel dimensions (nrows, ncolumns), (ny, nx)
        samp: numpy.ndarray shape=(2,) dtype=float
            row and column sampling rates (sx, sy)
        flt_size: float
            filter kernel size (HWHM) in physical units
    
    Return:
        kernel: numpy.ndarray shape=(nrows, ncolumns), dtype=float
    '''
    srcsz = np.abs(flt_size)
    kernel = np.zeros((ndim[0], ndim[1]), dtype=float)
    ndim2 = np.array([ndim[0] >> 1, ndim[1] >> 1])
    krad = 4 * srcsz # calculate filter kernel up to 4 sigma
    kr2 = krad**2
    nkx2 = int(min(np.ceil(krad / samp[0]),ndim2[1]))
    nky2 = int(min(np.ceil(krad / samp[1]),ndim2[0]))
    kprm = -np.log(2.) / srcsz**4
    for j in range(-nky2, nky2+1):
        j1 = j % ndim[0]
        ry = samp[1] * j
        ry2 = ry**2
        for i in range(-nkx2, nkx2+1):
            i1 = i % ndim[1]
            rx = samp[0] * i
            r2 = ry2 + rx**2
            if (r2 < kr2):
                kval = np.exp(kprm * r2**2)
                kernel[j1,i1] = kval
    return kernel
# %%
@jit
def filter_exp4_2d(ndim, samp, flt_size_a, flt_size_b, flt_dir):
    '''
    Returns a exponential forth-order anisotropic filter kernel for a low-pass in an array of
    given dimension. The filter is returned with value 1 at pixel [0,0]. Normalization should
    be done numerically.

    Parameters:
        ndim: numpy.ndarray shape=(2,) dtype=int
            kernel dimensions (nrows, ncolumns), (ny, nx)
        samp: numpy.ndarray shape=(2,) dtype=float
            row and column sampling rates (sx, sy)
        flt_size_a: float
            filter kernel size (HWHM) in physical units along orientation a
        flt_size_b: float
            filter kernel size (HWHM) in physical units along orientation b perpendicular to a
        flt_dir: float
            filter kernel a axis orientation to the grid row axis in radians
    
    Return:
        kernel: numpy.ndarray shape=(nrows, ncolumns), dtype=float
    '''
    sz_a = np.abs(flt_size_a)
    sz_b = np.abs(flt_size_b)
    sz_d = flt_dir
    kernel = np.zeros((ndim[0], ndim[1]), dtype=float)
    ndim2 = np.array([ndim[0] >> 1, ndim[1] >> 1])
    krad = 3 * np.amax([sz_a, sz_b]) # calculate filter kernel up to 3 sigma
    kr2 = krad**2
    nkx2 = int(min(np.ceil(krad / samp[0]), ndim2[1]))
    nky2 = int(min(np.ceil(krad / samp[1]), ndim2[0]))
    kprm = -np.log(2.)
    sdir = np.sin(sz_d)
    cdir = np.cos(sz_d)
    for j in range(-nky2, nky2+1):
        j1 = j % ndim[0]
        ry = samp[1] * j
        ry2 = ry**2
        py2prm = ry2 * (cdir**2 / sz_b**2 + sdir**2 / sz_a**2)
        for i in range(-nkx2, nkx2+1):
            i1 = i % ndim[1]
            rx = samp[0] * i
            rx2 = rx * rx
            px2prm = rx2 * (cdir**2 / sz_a**2 + sdir**2 / sz_b**2)
            pxyprm = 2. * rx * ry * (1./sz_a**2 - 1./sz_b**2) * sdir * cdir
            r2 = ry2 + rx2
            if (r2 < kr2):
                kval = np.exp( kprm * (px2prm + py2prm + pxyprm)**2 )
                kernel[j1,i1] = kval
    return kernel
# %%
@jit
def filter_sigmoid_2d(ndim, samp, flt_size, smoothness=0., anisotropy=0., orientation=0.):
    '''
    Returns a sigmodial anisotropic filter kernel for a low-pass in an array of given dimension.

    Parameters:
        ndim: numpy.ndarray shape=(2,) dtype=int
            kernel dimensions (nrows, ncolumns), (ny, nx)
        samp: numpy.ndarray shape=(2,) dtype=float
            row and column sampling rates (sx, sy)
        flt_size: float
            filter size (HWHM) in physical units
        smoothness: float
            edge smoothness of the kernel in physical units
        anisotropy: float
            kernel anisotropy relative to flt_size
        orientation: float
            anisotropy orientation of large axis to rows, ]-pi, pi]
    
    Return:
        kernel: numpy.ndarray shape=(nrows, ncolumns), dtype=float
    '''
    srcsz = np.abs(flt_size)
    kernel = np.zeros((ndim[0], ndim[1]), dtype=float)
    ndim2 = np.array([ndim[0] >> 1, ndim[1] >> 1])
    apert.aperture_a_grid(kernel,ndim2.astype(float),np.array([[samp[0],0.],[0.,samp[1]]]),np.array([0.,0.]),srcsz,smoothness,anisotropy,orientation)
    kernel[...] = np.roll(kernel, shift = ndim2, axis = (0,1))
    return kernel
# %%
def normal_1d(n, p0, sigma):
    """

    Returns a 1d array of length n with a normal distribution
    of r.m.s. width sigma centered a position p0.

    Parameters
    ----------
        n : int
            number of samples of the 1d output
        p0 : float
            center position of the normal distribution
        sigma : float
            r.m.s. width of the distribution

    Returns
    -------
        numpy.ndarray of length n

    Remarks
    -------
        periodic wrap around is assumed

    """
    assert n > 1, 'n must be larger than one'
    assert np.abs(sigma) > 0, 'sigma must be non-zero'
    pf = -0.5 / sigma**2
    n2 = n>>1 # index of the center pixel
    l_kern = np.zeros(n)
    n0 = int(np.floor(p0))
    p0_frac = p0 - n0
    sum_kern = 0.
    for i in range(0, n):
        x = 1. * (i - n2) + p0_frac
        v_kern = np.exp(pf * x**2)
        l_kern[i] = v_kern
        sum_kern += v_kern
    assert sum_kern > 0, 'intermediate kernel power is zero'
    return np.roll(l_kern / sum_kern, n0-n2)