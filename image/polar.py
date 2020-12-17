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
@jit
def polar_resample(image, num_rad, num_phi, pole, rng_rad, rng_phi = [0.,2. * np.pi], image_scale = [1., 1.], ipol = 1):
    """

    Transforms a 2D image to a polar grid using interpolation.

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
            radial range in units of the input grid scale
        rng_phi : float array of length 2
            azimuthal range in radian, use only values from the positive interval [0, 2*Pi]!
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
        * The routine as a @jit decorator and may require more time for compiling
          during the first call.

    """
    nd = image.shape
    # check input
    assert len(nd)==2, 'this works only for 2d arrays as input'
    assert num_rad>=1, 'number of radial samples must be at least 1'
    assert num_phi>=1, 'number of azimuthal samples must be at least 1'
    assert len(rng_rad)==2, 'invalid length of radial range input'
    assert len(rng_phi)==2, 'invalid length of azimuthal range input'
    assert (rng_rad[0]<rng_rad[1] and rng_rad[0]>=0.), 'invalid radial range parameter'
    assert len(image_scale)>=2, 'invalid length of input image scale parameter'
    assert (np.abs(image_scale[0])>0. and np.abs(image_scale[1])>0.), 'invalid input image scale'
    assert (ipol>=0 and ipol<=1), 'invalid interpolation switch (0,1)'
    # initialize
    ny = nd[0] # number of input rows (y)
    nx = nd[1] # number of input columns (x)
    r0 = rng_rad[0] # radial range start
    r1 = rng_rad[1] # radial range stop
    p0 = rng_phi[0] # azimuth range start
    p1 = rng_phi[1] # azimuth range stop
    if (p0 == p1): p1 = p0 + 2.*np.pi # same phi -> full circle
    image_out = np.zeros((num_rad,num_phi)) # polar data initialized to zero
    x0 = pole * image_scale # position of the pole in physical units of the input grid
    # loop over polar coordinates
    for ir in range(0, num_rad): # loop over radii
        r = r0 + (r1 - r0) * ir / num_rad # radial coordinate
        for ip in range(0, num_phi): # loop over azimuth
            p = p0 + (p1 - p0) * ip / num_phi # azimuthal coordinate
            x = x0 + r * np.array([np.cos(p), np.sin(p)]) # position on input grid scale
            pos = x / image_scale # position 
            if ((pos[0] >= 0) and (pos[0] <= nx-1) and (pos[1] >= 0) and (pos[1] <= ny-1)):
                image_out[ir,ip] = aimg.image_at(image, pos, ipol) # get from input image by interpolation
    return image_out
# %%
@jit
def polar_rebin(image, num_rad, num_phi, pole, rng_rad, rng_phi = [0.,2. * np.pi], image_scale = [1., 1.]):
    """

    Transforms a 2D image to new grid in polar representation using a re-binning algorithm.

    Parameters:
        image : numpy.array of 2 dimensions
            data on a regular grid
        num_rad : integer
            number of radial sample
        num_phi : integer
            number of azimuthal samples
        pole : float array of length 2
            position (x,y) of the pole in the input image (fractional pixel coordinate)
        rng_rad : float array of length 2
            radial range in pixels
        rng_phi :float array of length 2
            azimuthal range in radian, use only values from the positive interval [0, 2*Pi]!
            default = np.array((0.,2.*np.pi))
        image_scale : float array of length 2
            scale (x, y) of the input image in physical units per pixel, e.g. nm/pixel
            default = [1., 1.]

    Returns:
        numpy.array of 2 dimensions, shape (num_rad,num_phi)

    Remarks:
        * Due to the non-isotropic distribution of polar bins, some output bins
          may not receive data, especially around the pole region. If you want
          to avoid this, you may use the routine polar_resample.
        * If the scale argument is used, the rng_rad input is assumed to be meant
          on the same scale as the input scale, i.e. the same physical scale unit.
          Also the azimuth argument is assumed to address physical angles, which may
          deviate from angles of a regular grid.
        * The routine as a @jit decorator and may require more time for compiling
          during the first call.
        
    """
    nd = image.shape
    # check input
    assert len(nd)==2, 'this works only for 2d arrays as input'
    assert num_rad>=1, 'number of radial samples must be at least 1'
    assert num_phi>=1, 'number of azimuthal samples must be at least 1'
    assert len(rng_rad)==2, 'invalid length of radial range input'
    assert len(rng_phi)==2, 'invalid length of azimuthal range input'
    assert (rng_rad[0]<rng_rad[1] and rng_rad[0]>=0.), 'invalid radial range parameter'
    assert len(image_scale)>=2, 'invalid length of input image scale parameter'
    assert (np.abs(image_scale[0])>0. and np.abs(image_scale[1])>0.), 'invalid input image scale'
    # initialize
    tpi = 2.*np.pi # 2*Pi
    ny = nd[0]
    nx = nd[1]
    r0 = rng_rad[0]
    r1 = rng_rad[1]
    p0 = rng_phi[0]
    p1 = rng_phi[1]
    x0 = pole * image_scale # position of the pole in physical units of the input grid
    image_out = np.zeros((num_rad,num_phi)) # polar data
    check_phi = 0
    delta_p = p1 - p0 # input azimuth range
    if (np.abs(delta_p)>=tpi or np.abs(delta_p)==0.): # full circle
        delta_p = tpi
    else: # not a full circle
        check_phi = 1
    # initialize bounding box around the full outer circle
    ix0 = np.min([nx-1, np.max([0, int(np.floor((x0[0] - r1)/image_scale[0]))])]) # left pixel range clipped to image
    ix1 = np.min([nx-1, np.max([0, int(np.ceil((x0[0] + r1)/image_scale[0]))])]) # right pixel range clipped to image
    iy0 = np.min([ny-1, np.max([0, int(np.floor((x0[1] - r1)/image_scale[1]))])]) # bottom pixel range clipped to image
    iy1 = np.min([ny-1, np.max([0, int(np.ceil((x0[1] + r1)/image_scale[1]))])]) # top pixel range clipped to image
    # (complicated idea, re-think here!) if (check_phi == 1) and (delta_p / np.pi < 1.5): # adjust bounding box
    # accumulate polar samples to polar bins
    for j in range(iy0, iy1+1):
        dy = (j - pole[1]) * image_scale[1]
        dy2 = dy*dy
        for i in range(ix0, ix1+1):
            dx = (i - pole[0]) * image_scale[0]
            ir2 = dx*dx + dy2
            ir = np.sqrt(ir2)
            if (ir < r0 or ir > r1): continue # outside radial range, skip pixel
            ip = p0 # preset azimuth bin index to first index
            if (ir > 0.): # if r==0 this means, that all pixels will be accumulated by the first azimuth pixel
                ip = np.arctan2(dy,dx) % tpi # pixel azimuth -> into range [0, 2 pi]
                ipr = (ip - p0) / delta_p # azimuth relative to rng_phi - 0 ... 1
                if (check_phi): # segment
                    if (ipr < 0. or ipr > 1.): continue # outside azimuthal range, skip pixel
            jr = int(np.round((ir - r0) / (r1 - r0) * num_rad)) # round to next radial bin
            jp = int(np.round(ipr * num_phi)) # round to next azimuthal bin
            #print("i =",i,", j =",j,", r =",ir,", p =", ip,", jr =",jr,", jp =",jp)
            if (jr >= 0 and jr < num_rad and jp >= 0 and jp < num_phi):
                image_out[jr,jp] += image[j,i]
    return image_out
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
