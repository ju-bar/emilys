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

        numpy.ndarray (complex)
            real part is the x DPC component
            imaginary part is the y DPC component

    """
    nd = a.shape
    pf = -1.J / (2. * np.pi)
    kx = np.tile([np.fft.fftfreq(nd[1])],(nd[0],1))
    ky = np.tile([np.fft.fftfreq(nd[0])],(nd[1],1)).T
    k2 = kx**2 + ky**2
    k2[0,0] = 1.0
    ik2 = 1.0 / k2
    ik2[0,0] = 0.0
    kmin = max(1. / nd[0], 1. / nd[1])
    ft02 = max(kmin, rk[0])**2
    ft12 = max(kmin, rk[1])**2
    flt = (1. - np.exp(-1.0 * (k2 / ft02)**fp)) * np.exp(-1.0 * (k2 / ft12)**fp)
    z = dpc_image(a, b, c, d, alpha)
    fzx = np.fft.fft2(z.real)
    fzy = np.fft.fft2(z.imag)
    fidpc = pf * (kx * fzx + ky * fzy) * flt * ik2
    idpc = np.fft.ifft2(fidpc)
    #print(np.sum(idpc.real),np.sum(idpc.imag))
    return idpc.real




