# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:21:00 2020
@author: ju-bar

Routines for Thon ring analysis.

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
# %%
from numba import jit # include compilation support
import numpy as np
from emilys.image.polar import polar_resample, polar_radpol3_transform
from emilys.numerics.linsearch import linsrch_minloc1
from emilys.image.kernels import filter_normal_iso_2d, filter_normal_2d, filter_exp4_2d
from scipy.optimize import curve_fit
# %%
@jit
def lf_component(diffractogram):
    '''
    Calculates the low-frequency diffractogram component by adaptive filtering of the
    auto-correlation function. This should return an image, where Thon rings modulations
    are effectively filtered out, except for the first ring.

    Parameters:
        diffractogram : numpy.array of dimension 2
            input diffractogram data

    Result:
        numpy.array of the same size as diffractogram

    Remarks:
        1) The low-frequency component in the auto-correlation function is assumed to be smaller than 10% Nyquist.
        2) An anisotropic filter will be used adopting to the shape of the inner peak of the auto-correlation function.

    '''
    ndim = diffractogram.shape
    assert (len(ndim)==2), 'this is for 2d input arrays only'
    ndim2 = np.array([ndim[0]>>1,ndim[1]>>1]) # get nyquist index
    nxy = 0.5 * (ndim2[0] + ndim2[1])
    adifft = np.fft.ifft2(diffractogram)
    acf0 = np.abs(adifft)  # calculate the auto-correlation function
    acf1 = np.roll(acf0, shift = ndim2, axis = (0,1)) # recenter the acf
    # extract the polar transform of the central part
    crng = 1 + int(0.1 * nxy) # central range limiting low-frequency search
    apolc = polar_resample(acf1,crng,32,ndim2,np.array((0.,1.*crng)),np.array((0.,2*np.pi))).T
    # determine the extent of the central peak in the auto-correlation function
    lwcpk = linsrch_minloc1(apolc,False)
    # print(lwcpk)
    # calculate a low-pass filter kernel for extracting the diffractogram low-frequency part
    flt_rad_max = np.amax(lwcpk)
    flt_rad_min = np.amin(lwcpk)
    flt_dir = np.angle(np.fft.fft(lwcpk)[2]) + 0.5 * np.pi
    alpflt0 = filter_normal_2d(ndim,[1.,1.],flt_rad_max,flt_rad_min,flt_dir)
    # apply low-pass filtering
    return np.real(np.fft.fft2(adifft * alpflt0))
# %%
@jit
def lp_filter(diffractogram, freq):
    '''
    calculates a radial low-pass filter of the input with frequency
    limit freq in units of nyquist using a Gaussian filter function

    diffractogram : numpy.array of 2 dimensions
        input image / diffractogram
    freq : relative filter strength (0 ... 1)
    '''
    ndim = diffractogram.shape
    assert (len(ndim)==2), 'this is for 2d input arrays only'
    ndim2 = np.array([ndim[0]>>1,ndim[1]>>1]) # get nyquist index
    nxy = 0.5 * (ndim2[0] + ndim2[1])
    aft = np.fft.fft2(diffractogram)
    flt_rad = freq * nxy
    aff = filter_normal_iso_2d(ndim,[1.,1.],flt_rad)
    return np.fft.ifft2(aff * aft).real
