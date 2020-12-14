# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:19:00 2020
@author: ju-bar

Functions related to work with crystal structures

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
from emilys.optics.econst import ht2wl, k2theta, theta2qperp
# %%
def HOLZ_theta(e0, c, n=1, de=0., verbose=False):
    """

    Calculates the scattering angle of the nth-order Laue zone
    for a crystal with lattice constant c along the incident beam
    direction and for scattering with energy loss de.

    Parameters
    ----------

        e0 : float
            electron beam kinetic energy in eV
        c : float
            lattice constant along the beam direction in nm
        n : int
            order of the Laue zone, default: 1
        de : float  
            electron energy loss in eV, default: 0
        verbose : boolean
            flags extra output on (True) or off (False, default)

    Return
    ------
        theta : float
            scattering angle around which to expect HOLZ

    """
    assert n > 0, "this is not working for n < 1."
    theta = 0.
    if (np.abs(de) >= e0):
        print("Warning: energy not conserved, de is larger or equal e0.")
        return theta
    k0 = 1. / ht2wl(e0 * 0.001) # electron wave number (total momentum)
    if (verbose): print('k0 = {:.5E}'.format(k0))
    kp = 1. / ht2wl((e0-np.abs(de)) * 0.001) # electron wave number after scattering (by conservation of energy)
    if (verbose): print('kp = {:.5E}'.format(kp))
    kc = n / c # longitudinal reciprocal lattice vector length (from k0 to the Laue zone)
    if (verbose): print('kc = {:.5E}'.format(kc))
    kz = k0 - kc # residual vertical wave number (from center to Laue zone)
    if (verbose): print('kz = {:.5E}'.format(kz))
    if (kz > kp): # ... k' must be able to reach the Laue zone (forward scattering)
        print("Warning: momentum not conserved, kz is larger than k'")
        return theta
    if (kz < -kp): # ... k' must be able to reach the Laue zone (backward scattering)
        print("Warning: momentum not conserved, kz is smaller than -k'")
        return theta
    kt = np.sqrt(kp**2 - kz**2) # transversal component
    if (verbose): print('kt = {:.5E}'.format(kt))
    q = np.sqrt((kp-kz)**2 + kt**2) # scattering vector on the Laue circle
    if (verbose): print('q  = {:.5E}'.format(q))
    theta = k2theta(q, e0, de)
    return theta

def HOLZ_qperp(e0, c, n=1, de=0., verbose=False):
    """

    Calculates the perpendicular component of the scattering vector
    to the nth-order Laue zone for a crystal with lattice constant c
    along the incident beam direction and for scattering with energy
    loss de.

    Parameters
    ----------

        e0 : float
            electron beam kinetic energy in eV
        c : float
            lattice constant along the beam direction in nm
        n : int
            order of the Laue zone, default: 1
        de : float  
            electron energy loss in eV, default: 0
        verbose : boolean
            flags extra output on (True) or off (False, default)

    Return
    ------
        qperp : float
            perpendicular component of the scattering vector [1/nm]

    """
    return theta2qperp(HOLZ_theta(e0, c, n, de, verbose), e0, de)