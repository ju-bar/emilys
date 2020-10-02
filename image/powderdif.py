# -*- coding: utf-8 -*-
"""
Created on Wed Oct 02 17:02:00 2020
@author: ju-bar

Routines for powder diffraction data analysis.

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
# %%
from numba import jit # include compilation support
import numpy as np
from emilys.image.arrayplot import arrayplot2d
from matplotlib.patches import Circle
# %%
def get_DSrings_radi(ndim, hkl, a0, dpix, mag):
    '''

    Calculate the radius in pixels of Debye-Scherrer rings in a diffractogram of an image
    of a powder sample recorded with a given magnification. Assumes a square size image
    and a cubic lattice.

    Parameters:
        ndim : int
            size of the image in pixels
        hkl : array of shape (n,3) of type int
            list of [h,k,l] indices of a cubic reciprocal lattice
        a0 : float
            real lattice constant
        dpix : float
            image pixel size (same unit as a0)
        mag : float
            magnification factor of the image

    Returns:
        array (n) of floats
            list of radii of Debye-Scherrer rings in the 2D Fourier transform of the image.

    '''
    nring = len(hkl)
    lq = np.zeros(nring)
    lr = lq.copy()
    for i in range(0, nring): # generate list of diffraction vectors related to the lattice
        lq[i] = np.sqrt(np.dot(hkl[i],hkl[i])) / a0
        lr[i] = lq[i] * dpix * ndim / mag
    return lr
# %%
def plot_DSrings_on_dif(dif, hkl, a0, dpix, mag, psca = 0.5, pamp = 0.25, prng = [0.,0.5]):
    '''

    Plots a diffractogram (FT amplitudes of an image) and overlays rings corresponding to
    the reciprocal lattice vectors given by the list hkl for a cubic crystal lattice of
    lattice constant a0. An image pixel size dpix and magnification used to record the image
    must be given. pamp is an exponent used to display the intensities and prng is the
    relative range of intensities used for display scaling colors from min to max.

    '''
    ndim = dif.shape[0]
    ndim2 = ndim >> 1
    nring = len(hkl)
    lr = get_DSrings_radi(ndim, hkl, a0, dpix, mag)
    pdif = arrayplot2d(dif**pamp, psca, 'inferno', vrange = prng)
    porg = [ndim2, ndim2]
    print(lr)
    for i in range(0, nring):
        if (lr[i] < ndim2):
            circ = Circle(porg, lr[i], color='b', linestyle=(5, (10, 4)), fill=False)
            pdif[1].add_patch(circ)