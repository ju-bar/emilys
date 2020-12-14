# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 17:30:00 2020
@author: ju-bar

Polar image transformation

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
# %%
from numba import jit # include compilation support
import numpy as np # include numeric functions
import emilys.image.imagedata as aimg # include image access routines
# %%
#@jit
def polar_resample(image, num_rad, num_phi, pole, rng_rad, rng_phi = [0.,2. * np.pi], ipol = 1):
    '''

    Transforms a 2D image to new grid in polar representation using interpolation.

    Parameters:
        image : numpy.array of 2 dimensions
            data on a regular grid
        num_rad : integer
            number of radial sample
        num_phi : integer
            number of azimuthal samples
        pole : numpy.array, size=2
            position (x,y) of the pole in the input image (fractional pixel coordinate)
        rng_rad : numpy.array, size=2
            radial range in pixels
        rng_phi : numpy.array, size=2
            azimuthal range in radian, use only values from the positive interval [0, 2*Pi]!
            default = np.array((0.,2.*np.pi))
        ipol : int
            interpolation type
            0 = nearest neigbor
            1 = bi-linear

    Returns:
        numpy.array of 2 dimensions, shape (num_rad,num_phi)

    Remarks:
        May produce strange signal with noise data and of the polar sampling is fine
        compared to the input grid.
        This assumes isotropic sampling of the input.

    '''
    nd = image.shape
    # check input
    assert len(nd)==2, 'this works only for 2d arrays as input'
    assert num_rad>=1, 'number of radial samples must be at least 1'
    assert num_phi>=1, 'number of azimuthal samples must be at least 1'
    assert len(rng_rad)==2, 'invalid length of radial range input'
    assert len(rng_phi)==2, 'invalid length of azimuthal range input'
    assert (rng_rad[0]<rng_rad[1] and rng_rad[0]>=0.), 'invalid radial range parameter'
    assert (ipol>=0 and ipol<=1), 'invalid interpolation switch (0,1)'
    # initialize
    ny = nd[0]
    nx = nd[1]
    r0 = rng_rad[0]
    r1 = rng_rad[1]
    p0 = rng_phi[0]
    p1 = rng_phi[1]
    if (p0 == p1): p1 = p0 + 2.*np.pi # same phi -> full circle
    image_out = np.zeros((num_rad,num_phi)) # polar data
    # loop over polar coordinates
    for ir in range(0, num_rad): # loop over radii
        r = r0 + (r1 - r0) * ir / num_rad # radial coordinate
        for ip in range(0, num_phi): # loop over azimuth
            p = p0 + (p1 - p0) * ip / num_phi# azimuthal coordinate
            pos = pole + r * np.array([np.cos(p), np.sin(p)])
            if ((pos[0] >= 0) and (pos[0] <= nx-1) and (pos[1] >= 0) and (pos[1] <= ny-1)):
                image_out[ir,ip] = aimg.image_at(image, pos, ipol)
    return image_out
# %%
@jit
def polar_transform(image, num_rad, num_phi, pole, rng_rad, rng_phi = np.array((0.,2. * np.pi))):
    '''

    Transforms a 2D image to new grid in polar representation using a re-binning algorithm.

    Parameters:
        image : numpy.array of 2 dimensions
            data on a regular grid
        num_rad : integer
            number of radial sample
        num_phi : integer
            number of azimuthal samples
        pole : numpy.array, size=2
            position (x,y) of the pole in the input image (fractional pixel coordinate)
        rng_rad : numpy.array, size=2
            radial range in pixels
        rng_phi : numpy.array, size=2
            azimuthal range in radian, use only values from the positive interval [0, 2*Pi]!
            default = np.array((0.,2.*np.pi))

    Returns:
        numpy.array of 2 dimensions, shape (num_rad,num_phi)

    Remarks:
        Due to the re-binning algorithm, some bins may not receive data, especially when
        using fine sampling around the pole region. If you want to avoid this, you may use
        the routine polar_resample.
    '''
    nd = image.shape
    # check input
    assert len(nd)==2, 'this works only for 2d arrays as input'
    assert num_rad>=1, 'number of radial samples must be at least 1'
    assert num_phi>=1, 'number of azimuthal samples must be at least 1'
    assert len(rng_rad)==2, 'invalid length of radial range input'
    assert len(rng_phi)==2, 'invalid length of azimuthal range input'
    assert (rng_rad[0]<rng_rad[1] and rng_rad[0]>=0.), 'invalid radial range parameter'
    # initialize
    tpi = 2.*np.pi # 2*Pi
    ny = nd[0]
    nx = nd[1]
    r0 = rng_rad[0]
    r1 = rng_rad[1]
    p0 = rng_phi[0]
    p1 = rng_phi[1]
    #norm_out = np.zeros((num_rad,num_phi)) # polar bin population
    image_out = np.zeros((num_rad,num_phi)) # polar data
    check_phi = 0
    delta_p = p1 - p0 # input azimuth range
    if (np.abs(delta_p)>=tpi or np.abs(delta_p)==0.): # full circle
        delta_p = tpi
    else: # not a full circle
        check_phi = 1
    # determine bounding box around the annular segment added to the polar transform
    bx0 = np.min(np.array((r0 * np.cos(p0), r0 * np.cos(p1), r1 * np.cos(p0), r1 * np.cos(p1))) + pole[0]) # left-most corner of segment
    bx1 = np.max(np.array((r0 * np.cos(p0), r0 * np.cos(p1), r1 * np.cos(p0), r1 * np.cos(p1))) + pole[0]) # right-most corner of segment
    by0 = np.min(np.array((r0 * np.sin(p0), r0 * np.sin(p1), r1 * np.sin(p0), r1 * np.sin(p1))) + pole[1]) # lowest corner of segment
    by1 = np.max(np.array((r0 * np.sin(p0), r0 * np.sin(p1), r1 * np.sin(p0), r1 * np.sin(p1))) + pole[1]) # highest corner of segment
    # print("- b-box: ((", bx0, ",", by0, "),(", bx1, ",", by1, "))")
    if (p0 <= 0. and p1 >= 0.): bx1 = pole[0] + r1 # include x = x0 + r1
    if (p0 <= np.pi and p1 >= np.pi): bx0 = pole[0] - r1 # include x = x0 - r1
    if (p0 <= 0.5*np.pi and p1 >= 0.5*np.pi): by1 = pole[1] + r1 # include y = y0 + r1
    if (p0 <= 3.*np.pi/2. and p1 >= 3.*np.pi/2.): by0 = pole[1] - r1 # include y = y0 - r1
    # print("- b-box: ((", bx0, ",", by0, "),(", bx1, ",", by1, "))")
    ix0 = np.min([nx-1, np.max([0, int(np.floor(bx0))])]) # pixel range clipped to image
    ix1 = np.min([nx-1, np.max([0, int( np.ceil(bx1))])])
    iy0 = np.min([ny-1, np.max([0, int(np.floor(by0))])])
    iy1 = np.min([ny-1, np.max([0, int( np.ceil(by1))])])
    # print("- bounding box: ((", ix0, ",", iy0, "),(", ix1, ",", iy1, "))")
    # accumulate polar samples to polar bins
    for j in range(iy0, iy1+1):
        dy = 1.*(j - pole[1])
        dy2 = dy*dy
        for i in range(ix0, ix1+1):
            dx = 1.*(i - pole[0])
            ir2 = dx*dx + dy2
            ir = np.sqrt(ir2)
            if (ir < r0 or ir > r1): continue # outside radial range, skip pixel
            ip = p0
            if (ir > 0.): # 
                ip = np.arctan2(dy,dx) % tpi # pixel azimuth -> into range [0, 2 pi]
                ipr = (ip - p0) / delta_p # azimuth relative to rng_phi - 0 ... 1
                if (check_phi): # segment
                    if (ipr < 0. or ipr > 1.): continue # outside azimuthal range, skip pixel
            jr = int(np.round((ir - r0) / (r1 - r0) * num_rad)) # round to next radial bin
            jp = int(np.round(ipr * num_phi)) # round to next azimuthal bin
            #print("i =",i,", j =",j,", r =",ir,", p =", ip,", jr =",jr,", jp =",jp)
            if (jr >= 0 and jr < num_rad and jp >= 0 and jp < num_phi):
                #norm_out[jr,jp] += 1.
                image_out[jr,jp] += image[j,i]
    ## renormalize the output, removed (2020-10-13 JB)
    #for j in range(0, num_rad):
    #    for i in range(0, num_phi):
    #        if (norm_out[j,i]>0.): image_out[j,i] /= norm_out[j,i] # divide by number of accumulated samples to the polar bin
    return image_out
# # %%
# @jit
# def polar_transform_projector(nx, ny, ax, ay, nr, np, p0, rng_r, rng_p):
#     '''
    
#     Calculates a projection matrix from a recangular non-isotropic space grid to a polar grid.

#     Parameters:
#         nx : int
#             number of grid points along x (fast dimension)
#         ny : int
#             number of grid points along y (slow dimension)
#         ax : float
#             total grid size along x (physical unit)
#         ay : float
#             total grid size along y (physical unit)
#         nr : int
#             number of radial output samples (fast dimension)
#         np : int
#             number of azimuthal output samples (slow dimension)
#         p0 : [float,float]
#             position of the pole in the source array (x,y) (physical unit)
#         rng_r : [float,float]
#             radial range of the output (physical unit)
#         rng_p : [float,float]
#             azimuthal range of the output (radians)

#     Returns:
#         An array of dimension (ny, nx, 2) which assigns a tuple [ir, ip] of the
#         polar output to each pixel [iy, ix] of the cartesian input. 
    
#     Remarks:
#         The output projector p is used in the following way with in input a of
#         shape (ny, nx) and an output b of shape (nr, np):
#         for j in range(0,ny):
#             for i in range(0,nx):
#                 ir = p[j,i,0]
#                 ip = p[j,i,1]
#                 if (ir >= 0 and ip >= 0):
#                     b[ir,ip] += a[j,i]

#     '''
#     p = np.zeros((ny,nx,2), dytpe=np.int32) # projector matrix init (iy,ix) -> (ir,ip)
#     p -= 1 # init with -1
#     sy = ay / ny
#     sx = ax / nx
#     sr = (rng_r[1] - rng_r[0]) / nr
#     sp = (rng_p[1] - rng_p[0]) / np
#     for j in range(0, ny):
#         dy = sy * j - p0[1] # y distance to pole
#         dy2 = dy**2
#         for i in range(0, nx):
#             dx = sx * i - p0[0] # x distance to pole
#             dx2 = dx**2
#             r = np.sqrt(dx2 + dy2)
#             p = np.arctan(dy, dx)
#             ir = np.round((r - r0) / sr)
#             ip = np.round()
#     return p
# %%
@jit
def polar_radpol3_transform(image, num_rad, num_phi, pole, rng_rad, rng_phi = np.array((0.,2. * np.pi)), x0 = 0., x1 = 1., x2 = 0., x3 = 0.):
    '''

    Resamples a 2D image to new grid in polar representation with the radial direction
    sampling on a grid with x = x0 + x1 * r + x2 * r**2 + x3 + r**3, where r is the distance
    to the pole in the input image and x the coordinate of the output image along axis = 0.
    The resampled x axis will correspond to a finer step size towards outer regions of the
    input image causing the respective data to be distributed more sparsely in the output.

    Parameters:
        image : numpy.array of 2 dimensions
            data on a regular grid
        num_rad : integer
            number of radial sample
        num_phi : integer
            number of azimuthal samples
        pole : numpy.array, size=2
            position (x,y) of the pole in the input image (fractional pixel coordinate)
        rng_rad : numpy.array, size=2
            radial range in pixels
        rng_phi : numpy.array, size=2
            azimuthal range in radian, use only values from the positive interval [0, 2*Pi]!
            default = np.array((0.,2.*np.pi))
        x0 : float
            radial sampling offset, default = 0.
        x1 : float
            radial sampling linear coefficient, default = 1.
        x2 : float
            radial sampling 2nd order coefficient, default = 0.
        x3 : float
            radial sampling 3rd order coefficient, default = 0.

    '''
    nd = image.shape
    # check input
    assert len(nd)==2, 'this works only for 2d arrays as input'
    assert num_rad>=1, 'number of radial samples must be at least 1'
    assert num_phi>=1, 'number of azimuthal samples must be at least 1'
    assert len(rng_rad)==2, 'invalid length of radial range input'
    assert len(rng_phi)==2, 'invalid length of azimuthal range input'
    assert (rng_rad[0]<rng_rad[1] and rng_rad[0]>=0.), 'invalid radial range parameter'
    # initialize
    tpi = 2.*np.pi # 2*Pi
    ny = nd[0]
    nx = nd[1]
    r0 = rng_rad[0]
    r1 = rng_rad[1]
    xl0 = x0 + x1 * r0 + x2 * r0**2 + x3 * r0**3 # lower radial limit
    xl1 = x0 + x1 * r1 + x2 * r1**2 + x3 * r1**3 # upper radial limit
    p0 = rng_phi[0]
    p1 = rng_phi[1]
    # norm_out = np.zeros((num_rad,num_phi)) # polar bin population
    image_out = np.zeros((num_rad,num_phi)) # polar data
    check_phi = 0
    delta_p = p1 - p0 # input azimuth range
    if (np.abs(delta_p)>=tpi or np.abs(delta_p)==0.): # full circle
        delta_p = tpi
    else: # not a full circle
        check_phi = 1
    # determine bounding box around the annular segment added to the polar transform
    bx0 = np.min(np.array((r0 * np.cos(p0), r0 * np.cos(p1), r1 * np.cos(p0), r1 * np.cos(p1))) + pole[0]) # left-most corner of segment
    bx1 = np.max(np.array((r0 * np.cos(p0), r0 * np.cos(p1), r1 * np.cos(p0), r1 * np.cos(p1))) + pole[0]) # right-most corner of segment
    by0 = np.min(np.array((r0 * np.sin(p0), r0 * np.sin(p1), r1 * np.sin(p0), r1 * np.sin(p1))) + pole[1]) # lowest corner of segment
    by1 = np.max(np.array((r0 * np.sin(p0), r0 * np.sin(p1), r1 * np.sin(p0), r1 * np.sin(p1))) + pole[1]) # highest corner of segment
    # print("- b-box: ((", bx0, ",", by0, "),(", bx1, ",", by1, "))")
    if (p0 <= 0. and p1 >= 0.): bx1 = pole[0] + r1 # include x = x0 + r1
    if (p0 <= np.pi and p1 >= np.pi): bx0 = pole[0] - r1 # include x = x0 - r1
    if (p0 <= 0.5*np.pi and p1 >= 0.5*np.pi): by1 = pole[1] + r1 # include y = y0 + r1
    if (p0 <= 3.*np.pi/2. and p1 >= 3.*np.pi/2.): by0 = pole[1] - r1 # include y = y0 - r1
    # print("- b-box: ((", bx0, ",", by0, "),(", bx1, ",", by1, "))")
    ix0 = np.min([nx-1, np.max([0, int(np.floor(bx0))])]) # pixel range clipped to image
    ix1 = np.min([nx-1, np.max([0, int( np.ceil(bx1))])])
    iy0 = np.min([ny-1, np.max([0, int(np.floor(by0))])])
    iy1 = np.min([ny-1, np.max([0, int( np.ceil(by1))])])
    # print("- bounding box: ((", ix0, ",", iy0, "),(", ix1, ",", iy1, "))")
    # accumulate polar samples to polar bins
    for j in range(iy0, iy1+1):
        dy = 1.*(j - pole[1])
        dy2 = dy*dy
        for i in range(ix0, ix1+1):
            dx = 1.*(i - pole[0])
            ir2 = dx*dx + dy2
            ir = np.sqrt(ir2)
            if (ir < r0 or ir > r1): continue # outside radial range, skip pixel
            ip = p0
            if (ir > 0.): # no angle at r = 0
                ip = np.arctan2(dy,dx) % tpi # pixel azimuth -> into range [0, 2 pi]
                ipr = (ip - p0) / delta_p # azimuth relative to rng_phi - 0 ... 1
                if (check_phi): # segment
                    if (ipr < 0. or ipr > 1.): continue # outside azimuthal range, skip pixel
            x = x0 + x1 * ir + x2 * ir**2 + x3 * ir**3 # corresponding output grid coordinate
            jr = int(np.round((x - xl0) / (xl1 - xl0) * num_rad)) # round to next radial bin
            jp = int(np.round(ipr * num_phi)) # round to next azimuthal bin
            #print("i =",i,", j =",j,", r =",ir,", p =", ip,", jr =",jr,", jp =",jp)
            if (jr >= 0 and jr < num_rad and jp >= 0 and jp < num_phi):
                # norm_out[jr,jp] += 1.
                image_out[jr,jp] += image[j,i]
    # renormalize the output # removed (2020-10-13 JB)
    #for j in range(0, num_rad):
    #    for i in range(0, num_phi):
    #        if (norm_out[j,i]>0.): image_out[j,i] /= norm_out[j,i] # divide by number of accumulated samples to the polar bin
    return image_out
# %%
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
