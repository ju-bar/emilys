# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 10:55:00 2020
@author: ju-bar

Image data manipulation routines

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
# %%
from numba import jit # include compilation support
import numpy as np # include numeric functions
# %%
def replace_area_from_random_surrounding(image, pos, r1, r2):
    '''

    Replaces image data at point pos within radius r1 from randomly
    chosen data point in range r1 to r2.
    A modified copy of image is returned, image is not changed.

    Parameters:
        image : numpy.array of 2 dimensions
            data on a regular grid
        pos : numpy.array of 1 dimension
            2-dimensional coordinate (x,y)
        r1 : float
            radius of replacement destinations around pos
        r2 : float
            radius of replacement sources around pos
    '''
    ndim = image.shape
    assert len(ndim)==2, 'this works only with a 2d image'
    assert len(pos)==2, 'pos must specify a 2d position'
    assert (pos[0] >= 0 and pos[0] <= ndim[1]-1 and pos[1] >= 0 and pos[1] <= ndim[0]-1), 'pos must be inside the image'
    assert r2 > 1., 'r2 must be larger than 1'
    assert r1 + 1 < r2, 'r2 must be at least 1 pixel larger than r1'
    assert (r2 < ndim[0]/2 and r2 < ndim[1]/2), 'r2 must be smaller than image half size'
    # init
    image_out = image.copy()
    # determine bounding box of replacements to avoid looping through the whole image
    rpbx0 = max(0,int(np.floor(pos[0]-r1)))
    rpbx1 = min(ndim[1]-1,int(np.ceil(pos[0]+r1)))
    rpby0 = max(0,int(np.floor(pos[1]-r1)))
    rpby1 = min(ndim[0]-1,int(np.ceil(pos[1]+r1)))
    # loop over bounding box and replace all pixels with distances smaller or equal r1 from pos
    for j in range(rpby0, rpby1+1):
        dy = j - pos[1]
        dy2 = dy * dy
        for i in range(rpbx0, rpbx1+1):
            dx = i - pos[0]
            r = np.sqrt(dx * dx + dy2)
            if (r > r1): continue # outside r1, skip pixel
            # current pixel value will be replaced
            # dice random replacement until one is found in the image
            while True:
                srnd = np.random.rand(2)
                sprad = srnd[0] * (r2 - r1) + r1
                spphi = srnd[1] * 2. * np.pi
                spi = int(np.round(pos[0] + sprad * np.cos(spphi)))
                spj = int(np.round(pos[1] + sprad * np.sin(spphi)))
                if (spi >= 0 and spi < ndim[1] and spj >= 0 and spj < ndim[0]): break # replacement source found
            # do the replacement
            image_out[j,i] = image[spj,spi]
    # done, return modified copy
    return image_out
