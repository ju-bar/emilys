# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:43:00 2022
@author: ju-bar

Differential phase contrast routines

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

import numpy as np

def dpc_image(a, b, c, d, alpha = 0.0):
    """

    Calculates DPC images from for segmented STEM images a, b, c, d
    under rotation alpha in the diffraction plane.

    Parameters
    ----------

        a, b, c, d : numpy.ndarray, (dimension = 2)
            segmented STEM images
        alpha : float
            rotation angle [radians]

    Returns
    -------

        numpy.ndarray (complex)
            real part is the x DPC component
            imaginary part is the y DPC component

    """
    dx = (a - c).astype(np.float64)
    dy = (b - d).astype(np.float64)
    z = np.empty(dx.shape, dtype=np.complex128)
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    z.real = dx * ca - dy * sa
    z.imag = dy * ca + dx * sa
    return z

def idpc_image(a, b, c, d, alpha=0., rk = [0., 1.], fp = 2.):
    """

    Calculates an iDPC image from for segmented STEM images a, b, c, d
    under rotation alpha in the diffraction plane and band-pass filter.

    Parameters
    ----------

        a, b, c, d : numpy.ndarray, (dimension = 2)
            segmented STEM images
        alpha : float, default 0.
            rotation angle [radians]
        rk : list (len=2), default [0., 1.]
            band pass filter threshold in fractions of the
            image Nyquist frequency
        fp : float, default 2.
            frequency filter exponential power
            formula: exp(-1.0 * (k2 / thr)**fp)

    Returns
    -------

        numpy.ndarray (float)
            iDPC image

    """
    nd = a.shape # get image size
    pf = -1.J / (2. * np.pi) # setup imaginary prefactor 2 Pi J
    kx = np.tile([np.fft.fftfreq(nd[1])],(nd[0],1)) # get horizontal frequency list, tile to full grid
    ky = np.tile([np.fft.fftfreq(nd[0])],(nd[1],1)).T # get vertical frequency list, tile to full grid
    k2 = kx**2 + ky**2 # get list of squared frequency, on full grid
    k2[0,0] = 1.0 # set factor 1 on DC value to avoid division by zero below
    ik2 = 1.0 / k2 # inverse squared frequencies on full grid
    ik2[0,0] = 0.0 # reset DC inverse frequency to zero (no iDPC coefficient available for this)
    kmin = max(1. / nd[0], 1. / nd[1]) # get frequency steps
    ft02 = max(kmin, rk[0])**2 # setup lower bwl of filter from input rk
    ft12 = max(kmin, rk[1])**2 # setup upper bwl of filer from input rk
    flt = (1. - np.exp(-1.0 * (k2 / ft02)**fp)) * np.exp(-1.0 * (k2 / ft12)**fp) # calculate bwl filter
    z = dpc_image(a, b, c, d, alpha) # calculate dpc images (as complex quantity)
    fzx = np.fft.fft2(z.real) # Fourier transform DPCx
    fzy = np.fft.fft2(z.imag) # Fourier transform DPCy
    fidpc = pf * (kx * fzx + ky * fzy) * flt * ik2 # generate Fourier transform of the iDPC image incl. filter
    idpc = np.fft.ifft2(fidpc) # inverse FFT to obtain the iDPC data
    #print(np.sum(idpc.real),np.sum(idpc.imag))
    return idpc.real # return iDPC image as real part of the previous




