# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 17:30:00 2020
@author: ju-bar

Polar image transformation

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
#
import numba # include compilation support
import numpy as np # include numeric functions
import emilys.image.imagedata as aimg # include image access routines
from emilys.numerics.roots import root_poly_2 # include root finder for square equations
from timeit import default_timer as timer
#
def to_polar_2d(x):
    """

    For input tuples x = [x1,x2] calculates radius and azimuth [r,phi]
    with r = (x1**2 + x2**2)**0.5 and phi = arctan2(x2,x1)
    
    Parameters
    ----------
        x : tuple of floats
            cartesian x1, x2 coordinates
    
    Returns
    -------
        rp : tuple of floats
            [r, phi] 

    """
    return np.array([np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])])
#
def from_polar_2d(r, phi, x, y):
    """

    For input tuples [r,phi] calculates cartesian positions [x1,x2]
    with x1 = r * cos(phi), x2 = r * sin(phi)
    
    Parameters
    ----------
        r : float, input
            polar radius
        phi : float, input
            polar aziumth
        x : float, output
            cartesian x coordinate
        y : float, output
            cartesian y coordinate
    
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
#
@numba.jit
def polar_resample_core(img_xy, x0, y0, dx, dy, img_pr, r0, p0, dr, dp, ipol):
    """

    Polar resampling algorithm with simplified interface for the numba JIT compiler.
    
    Parameters
    ----------
        img_xy : numpy.ndarray(shape=(ny,nx),dtype=float)
            input image
        x0 : float
            x-origin of the cartesian coordinates in input grid pixels
        y0 : float
            y-origin of the cartesian coordinates in input grid pixels
        dx : float
            x-sampling rate of the cartesian coordinates in input grid pixels
        dy : float
            y-sampling rate of the cartesian coordinates in input grid pixels
        img_pr : numpy.ndarray(shape=(nr,np),dtype=float)
            output image
        r0 : float
            start radius in from the cartesian origin in scaled units
        p0 : float
            start azimuth from the cartesian x-axis in radians
        dr : float
            step size of the radial axis in the output in scaled units
        dp : float
            step size of the azimuth axis in the output in radians
        ipol : int
            interpolation switch
            0 : nearest neighbor
            1 : bi-linear
    """
    nyx = img_xy.shape
    #print("polar_resamp_core: ndim1:", nyx)
    nrp = img_pr.shape
    #print("polar_resamp_core: ndim2:", nrp)
    # initialize
    x_0 = x0 * dx
    y_0 = y0 * dy
    # loop over polar coordinates
    for ir in range(0, nrp[0]): # loop over radii
        r = r0 + dr * ir # radial coordinate
        for ip in range(0, nrp[1]): # loop over azimuth
            p = p0 + dp * ip # azimuthal coordinate
            ix = (x_0 + r * np.cos(p)) / dx # x pixel position on input grid
            iy = (y_0 + r * np.sin(p)) / dy # y pixel position on input grid
            if ((ix >= 0.) and (ix <= nyx[1]-1.) and (iy >= 0.) and (iy <= nyx[0]-1.)):
                pos = np.array([ix, iy])
                # img_pr[ir,ip] = aimg.image_at(img_xy, pos, ipol) # get from input image by interpolation
                if (ipol == 1): # bilinear interpolation
                    il = int(np.floor(pos[0])) # x
                    ih = il + 1
                    fi = pos[0] - il
                    jl = int(np.floor(pos[1])) # y
                    jh = jl + 1
                    fj = pos[1] - jl
                    v  = img_xy[jl,il] * (1-fj) * (1-fi)
                    v += img_xy[jl,ih] * (1-fj) * fi
                    v += img_xy[jh,il] * fj * (1-fi)
                    v += img_xy[jh,ih] * fj * fi
                    img_pr[ir,ip] = v
                else: # fall-back to nearest neighbor interpolation
                    jx = int(np.round(ix))
                    jy = int(np.round(iy))
                    img_pr[ir,ip] = img_xy[jy,jx]
    return img_pr
#
def polar_resample(image, num_rad, num_phi, pole, rng_rad, rng_phi = [0.,2. * np.pi], image_scale = [1., 1.], ipol = 1):
    """

    Transforms 2D images to polar grids using interpolation.

    Parameters
    ----------
        image : numpy.array of 2 dimensions
            data on a regular grid
        num_rad : integer
            number of radial sample
        num_phi : integer
            number of azimuthal samples
        pole : float array of length 2
            position (x,y) of the pole/origin in the input image (fractional pixel coordinate)
        rng_rad : float array of length 2
            radial range start and coverage (r_0, delta_r)
        rng_phi : float array of length 2
            azimuthal range start and coverage (phi_0, delta_phi) in radians
            default = [0., 2.*np.pi]
        image_scale : float array of length 2
            scale (x, y) of the input image in physical units per pixel, e.g. nm/pixel
            default = [1., 1.]
        ipol : int
            interpolation type
            0 = nearest neigbor
            1 = bi-linear

    Returns
    -------
        numpy.array of 2 dimensions, shape (num_rad,num_phi)

    Remarks
    -------
        * May produce strange signal with noisy data and if the polar sampling
          is too fine compared to the input grid. The latter usually happens
          around the pole.
        * Non-isotropic sampling of the input is optional via the scale argument.
        * If the scale argument is used, the rng_rad input is assumed to be meant
          on the same scale as the input scale, i.e. the same physical scale unit.
          Also the azimuth argument is assumed to address physical angles, which may
          deviate from angles of a regular grid.
        * Due to the non-isotropic sampling of a polar system, the norm of the
          input will not be conserved in the output. If this is important, you
          may want to try a re-binning algorithm such as in polar_rebin below.
        * The routine uses an interpolation wth @jit decorator. This may require
          more time for compiling during the first call.

    """
    nd = image.shape
    assert len(nd)>=2, 'need at least a 2d array'
    ld = len(nd)
    nx = nd[ld-1]
    ny = nd[ld-2]
    nimg = int(image.size / nx / ny)
    img = image.reshape(nimg, ny, nx) # reshape to a sequence of 2d images
    assert num_rad>=1, 'number of radial samples must be at least 1'
    assert num_phi>=1, 'number of azimuthal samples must be at least 1'
    npol = int(np.array(pole).size / 2)
    p_org = np.array(pole).reshape(npol,2) # sequence of pole positions
    assert (npol==nimg or npol==1), 'invalid number of pole positions'
    assert len(rng_rad)==2, 'invalid length of radial range input'
    assert np.abs(rng_rad[1])>0., 'invalid radial range parameter'
    assert len(rng_phi)==2, 'invalid length of azimuthal range input'
    assert np.abs(rng_phi[1])>0., 'invalid azimuth range parameter'
    assert len(image_scale)>=2, 'invalid length of input image scale parameter'
    assert (np.abs(image_scale[0])>0. and np.abs(image_scale[1])>0.), 'invalid input image scale'
    assert (ipol>=0 and ipol<=1), 'interpolation switch {:d} not supported'.format(ipol)
    # initialize
    r_0 = rng_rad[0]
    d_r = rng_rad[1] / num_rad
    p_0 = rng_phi[0]
    d_p = rng_phi[1] / num_phi
    d_x = image_scale[0]
    d_y = image_scale[1]
    image_out = np.zeros((nimg,num_rad,num_phi)) # polar data
    x_0 = p_org[0,0]
    y_0 = p_org[0,1]
    for i in range(nimg):
        if (npol>1):
            x_0 = p_org[i,0]
            y_0 = p_org[i,1]
        polar_resample_core(img[i,:,:], x_0, y_0, d_x, d_y, image_out[i,:,:], r_0, p_0, d_r, d_p,ipol)
    nout = np.array(nd).astype(int)
    nout[ld-2] = num_rad
    nout[ld-1] = num_phi
    return image_out.reshape(nout)
#
@numba.jit
def polar_rebin_core(img_xy, x0, y0, dx, dy, img_pr, r0, p0, dr, dp):
    """

    Polar rebinning algorithm with simplified interface for the numba JIT compiler.
    
    Parameters
    ----------
        img_xy : numpy.ndarray(shape=(ny,nx),dtype=float)
            input image
        x0 : float
            x-origin of the cartesian coordinates in input grid pixels
        y0 : float
            y-origin of the cartesian coordinates in input grid pixels
        dx : float
            x-sampling rate of the cartesian coordinates in input grid pixels
        dy : float
            y-sampling rate of the cartesian coordinates in input grid pixels
        img_pr : numpy.ndarray(shape=(nr,np),dtype=float)
            output image
        r0 : float
            start radius in from the cartesian origin in scaled units
        p0 : float
            start azimuth from the cartesian x-axis in radians
        dr : float
            step size of the radial axis in the output in scaled units
        dp : float
            step size of the azimuth axis in the output in radians
        
    """
    tpi = 2. * np.pi
    nyx = img_xy.shape
    #print("polar_rebin_core: ndim1:", nyx)
    nrp = img_pr.shape
    #print("polar_rebin_core: ndim2:", nrp)
    r1 = r0 + nrp[0] * dr
    p1 = p0 + nrp[1] * dp
    #print("polar_rebin_core: radial: ", [r0, r1, dr])
    #print("polar_rebin_core: azimuth: ", [p0, p1, dp])
    ix0 = min(nyx[1]-1, max(0, int(np.floor((x0 * dx - r1) / dx)))) # left pixel range clipped to image
    ix1 = min(nyx[1]-1, max(0, int(np.ceil((x0 * dx + r1) / dx)))) # right pixel range clipped to image
    iy0 = min(nyx[0]-1, max(0, int(np.floor((y0 * dy - r1) / dy)))) # bottom pixel range clipped to image
    iy1 = min(nyx[0]-1, max(0, int(np.ceil((y0 * dy + r1) / dy)))) # top pixel range clipped to image
    #print("polar_rebin_core: roi:", [[ix0,iy0],[ix1,iy1]])
    r = 0.
    phi = 0.
    for j in range(iy0, iy1+1):
        y = (j - y0) * dy
        ysqr = y * y
        for i in range(ix0, ix1+1):
            x = (i - x0) * dx
            r = np.sqrt(x * x + ysqr)
            if (r < r0 or r > r1): continue # skip pixels out of radial range
            phi = np.arctan2(y, x)
            if phi < 0: phi += tpi # bring phi to positive range
            if (phi < p0 or phi > p1): continue # ot of azimuth range
            jr = int(np.round((r - r0) / dr)) # round to next radial bin
            jp = int(np.round((phi - p0) / dp)) # round to next azimuthal bin
            if (jr >= 0 and jr < nrp[0] and jp >= 0 and jp < nrp[1]):
                img_pr[jr,jp] += img_xy[j,i]
#
def polar_rebin(image, num_rad, num_phi, pole, rng_rad, rng_phi = [0., 2.*np.pi], image_scale = [1., 1.]):
    """

    Transforms a 2D images to new grid in polar representation using a re-binning algorithm.

    Parameters
    ----------
        image : numpy.array floats with shape (...,ny,nx)
            data on a regular grid
        num_rad : integer
            number of radial sample
        num_phi : integer
            number of azimuthal samples
        pole : 2d tuples
            positions (x,y) of the pole in the input image (fractional pixel coordinate)
        rng_rad : float array of length 2
            radial range start and coverage (r_0, delta_r)
        rng_phi : float array of length 2
            azimuthal range start and coverage (phi_0, delta_phi) in radians
            default = [0., 2.*np.pi]
        image_scale : float array of length 2
            scale (x, y) of the input image in physical units per pixel, e.g. nm/pixel
            default = [1., 1.]

    Returns
    -------
        numpy.array of dimensions, shape (...,num_rad,num_phi)

    Remarks
    -------
        * Can be used to process on the last two dimensions of an array of
          higher dimensions.
        * Due to the non-isotropic distribution of polar bins, some output bins
          may not receive data, especially around the pole region. If you want
          to avoid this, you may use the routine polar_resample.
        * If the scale argument is used, the rng_rad input is assumed to be meant
          on the same scale as the input scale, i.e. the same physical scale unit.
          Also the azimuth argument is assumed to address physical angles, which may
          deviate from angles of a regular grid.
        * Calls a subroutine with @jit decorator. This does require longer excecution on
          the first call for run-time compiling, but will be faster afterwards.
        
    """
    nd = image.shape
    assert len(nd)>=2, 'need at least a 2d array'
    ld = len(nd)
    nx = nd[ld-1]
    ny = nd[ld-2]
    nimg = int(image.size / nx / ny)
    img = image.reshape(nimg, ny, nx) # reshape to a sequence of 2d images
    assert num_rad>=1, 'number of radial samples must be at least 1'
    assert num_phi>=1, 'number of azimuthal samples must be at least 1'
    npol = int(np.array(pole).size / 2)
    p_org = np.array(pole).reshape(npol,2) # sequence of pole positions
    assert (npol==nimg or npol==1), 'invalid number of pole positions'
    assert len(rng_rad)==2, 'invalid length of radial range input'
    assert np.abs(rng_rad[1])>0., 'invalid radial range parameter'
    assert len(rng_phi)==2, 'invalid length of azimuthal range input'
    assert np.abs(rng_phi[1])>0., 'invalid azimuth range parameter'
    assert len(image_scale)>=2, 'invalid length of input image scale parameter'
    assert (np.abs(image_scale[0])>0. and np.abs(image_scale[1])>0.), 'invalid input image scale'
    # initialize
    r_0 = rng_rad[0]
    d_r = rng_rad[1] / num_rad
    p_0 = rng_phi[0]
    d_p = rng_phi[1] / num_phi
    d_x = image_scale[0]
    d_y = image_scale[1]
    image_out = np.zeros((nimg,num_rad,num_phi)) # polar data
    x_0 = p_org[0,0]
    y_0 = p_org[0,1]
    for i in range(nimg):
        if (npol>1):
            x_0 = p_org[i,0]
            y_0 = p_org[i,1]
        polar_rebin_core(img[i,:,:], x_0, y_0, d_x, d_y, image_out[i,:,:], r_0, p_0, d_r, d_p)
    nout = np.array(nd).astype(int)
    nout[ld-2] = num_rad
    nout[ld-1] = num_phi
    return image_out.reshape(nout)
#
@numba.jit
def polar_rebin_rpoly3(img_xy, px0, py0, img_pr, r0, p0, dr, dp, x0, x1, x2, x3):
    """

    Transforms a 2D image to new grid in polar representation with the radial direction
    sampling on a grid with x = x0 + x1 * r + x2 * r**2 + x3 + r**3, where r is the distance
    to the pole in the input image and x the coordinate of the output image along axis = 0.
    The new x axis will correspond to a finer step size towards outer regions of the
    input image causing the respective data to be distributed more sparsely in the output.
    The transformation is based on a rebinning algorithm, which may lead to empty bins if
    the output grid has a higher density than the input grid.

    Parameters
    ----------
        img_xy : numpy.array of 2 dimensions, type float
            data on a regular grid
        px0, py0 : float
            position of the pole in the input image (fractional pixel coordinates)
        img_pr : numpy.array of 2 dimensions, type float
            outout polar grid
        r0 : float
            radial grid offset in input
        p0 : float
            azimuthal grid offset in input
        dr : float
            radial grid step size in input
        dp : float
            azimuthal grid step size in input
        x0 : float
            radial sampling offset, default = 0.
        x1 : float
            radial sampling linear coefficient, default = 1.
        x2 : float
            radial sampling 2nd order coefficient, default = 0.
        x3 : float
            radial sampling 3rd order coefficient, default = 0.
    
    Remarks
    -------
        * This routine has a @jit decorator. It does require longer excecution on
          the first call for run-time compiling, but will be faster afterwards.

    """
    nyx = img_xy.shape
    nrp = img_pr.shape
    tpi = 2.*np.pi # 2*Pi
    r1 = r0 + nrp[0] * dr
    p1 = p0 + nrp[1] * dp
    xl0 = x0 + x1 * r0 + x2 * r0**2 + x3 * r0**3 # lower radial limit
    xl1 = x0 + x1 * r1 + x2 * r1**2 + x3 * r1**3 # upper radial limit
    dxl = (xl1 - xl0) / nrp[0] # x sampling steps produced on the radial output dimension
    ix0 = min(nyx[1]-1, max(0, int(np.floor(px0 - r1)))) # left pixel range clipped to image
    ix1 = min(nyx[1]-1, max(0, int(np.ceil(px0 + r1)))) # right pixel range clipped to image
    iy0 = min(nyx[0]-1, max(0, int(np.floor(py0 - r1)))) # bottom pixel range clipped to image
    iy1 = min(nyx[0]-1, max(0, int(np.ceil(py0 + r1)))) # top pixel range clipped to image
    #
    # accumulate polar samples to polar bins
    r = 0.
    phi = 0.
    for j in range(iy0, iy1+1): # loop over input grid rows
        y = j - py0
        ysqr = y * y
        for i in range(ix0, ix1+1): # loop over input grid columns
            x = i - px0
            r = np.sqrt(x * x + ysqr)
            if (r < r0 or r > r1): continue # outside radial range, skip pixel
            phi = np.arctan2(y, x)
            if phi < 0: phi += tpi # wrap to range [0, 2 pi]
            if (phi < p0 or phi > p1): continue # out of azimuth range
            xl = x0 + x1 * r + x2 * r**2 + x3 * r**3 # corresponding output grid coordinate
            jr = int(np.round((xl - xl0) / dxl)) # round to next radial bin
            jp = int(np.round((phi - p0) / dp)) # round to next azimuthal bin
            if (jr >= 0 and jr < nrp[0] and jp >= 0 and jp < nrp[1]):
                img_pr[jr,jp] += img_xy[j,i]
    return
#
def radial_bin_mask(ndim, step, pix_pole, nr, radial_range=None):
    """

    Creates a mask linking image pixels to radial bins.

    Parameters
    ----------
        ndim : array of length 2 and type int
            dimensions of input images (num_rows, num_columns)
        step : array of length 2 and type float
            step sizes of input image (step_rows, step_columns)
        pix_pole : array of length 2 and type float
            pixel position of the pole from measuring radius (row, column)
        nr : int
            number of radial samples
        radial_range : array of lenth 2 and type float
            radial range, if None a default [0, nr] is used

    Returns
    -------
        array dimension ndim and type int containing indices of
        the radial 1d array linked to each input image

    """
    #
    msk = np.zeros(ndim, dtype=int)
    #
    if radial_range is None:
        r0 = 0.
        r1 = 1.0 * nr
    else:
        r0 = radial_range[0]
        r1 = radial_range[1]
    #
    for i_y in range(0, ndim[0]):
        y = step[0] * (i_y - pix_pole[0])
        for i_x in range(0, ndim[1]):
            x = step[1] * (i_x - pix_pole[1])
            r = np.sqrt(x**2 + y**2)
            i_r = round((r - r0) / (r1 - r0) * nr)
            if (i_r >= 0) and (i_r < nr):
                msk[i_y,i_x] = i_r
            else:
                msk[i_y,i_x] = -1
    #
    return msk
