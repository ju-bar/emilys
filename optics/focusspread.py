# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:16:00 2020
@author: ju-bar

Functions related to optical focus spread

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
# %%
def dist_gaussian(x, delta):
    '''

    Calculates values of a Gaussian focus spread distribution with
    focus spread 1/e-half-width delta.

    Parameters:
        x : float
            defocus value
        delta : float
            1/e-half-width of the focal distribution

    Returns:
        float

    '''
    return np.exp(-x**2 / delta**2) / (np.sqrt(np.pi) * np.abs(delta))

# %%
def info_limit(delta, lamb):
    '''

    Calculates the information limit for a Gaussian focus-spread with
    1/e half-width delta and for electrons of wavelength lamb.

    Parameters:
        delta : float
            focus spread 1/e-half-width
        lamb : float
            electron wave-length

    Returns:
        float : info-limit (same unit as delta and lamb)

    '''
    return ((np.pi*lamb*delta)**2 / 8.)**0.25
# %%
def delta_of_kappa(kappa, g, t, lamb):
    '''
    Calculate the focus-spread parameter from the circ_lf model
    parameter kappa for a given reciprocal space circle radius g
    beam tilt magnitude t and electron wavelength lamb
    '''
    return np.sqrt(0.5 * kappa) / (np.pi * lamb * g * t)
# %%
def kappa_of_delta(delta, g, t, lamb):
    '''
    Calculate the kappa parameter for the circ_lf model from
    focus spread delta for a given reciprocal space circle radius g
    beam tilt magnitude t and electron wavelength lamb
    '''
    return 2.0 * (np.pi * lamb * g * t * delta)**2

# %%
def circ_lf(phi, phi_t, g, t, c0, c1, kappa):
    '''
    
    Calculates a value of the low-frequency diffractogram component
    under tilted beam illumination in polar notation for points (g, phi) 
    in the diffractogram, beam tilts (t, phi_t), assuming an additional
    base line c0, a modulation amplitude c1 and focus spread dependent
    parameter kappa (see delta_of_kappa).
    The model parameters c0, c1 and kappa are usually not known and
    can be determined from values of the low-frequency content scanned
    along circles around the origin.

    Parameters:
        phi : float
            diffractogram point azimuth in radians
        phi_t : float
            beam tilt azimuth in radians
        g : float
            circle spatial frequency
        t : float
            beam tilt magnitude in units of a spatial frequency
        c0 : float
            base line value
        c1 : float
            contrast amplitude
        kappa : float
            width parameter depending on focus spread
    
    Returns: float

    '''
    return c0 + c1 * np.exp(-kappa * np.cos(phi - phi_t)**2) * np.cosh(kappa * g / t * np.cos(phi - phi_t))