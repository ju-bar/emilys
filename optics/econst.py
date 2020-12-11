# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 08:53:00 2020
@author: ju-bar

Constants and functions related to electrons and their physics

Declaration of physical constants based on published CODATA 2014
Source:	<http://dx.doi.org/10.1103/RevModPhys.88.035009>

Constant symbols are declared with capital letters.
Prefixes:
PHYS_ : general physics constants
EL_ : electron related data

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
# Exact constants
PHYS_C = 299792458. # vacuum speed of light [ m s^(-1) ]
PHYS_MU0 = 1.2566370614359172953850573533118E-6 # magnetic constant = 4*Pi * 10^(-7) [ N A^(-1) ]
PHYS_EPS0 =	8.8541878176203898505365630317108E-12 # electric constant = 1/(_MU0 _C0^2) [ F m^(-1) ]

# Derived constants
PHYS_HPL = 6.626070040E-34 # Planck's constant, error (81) in last digits [ J s ]
PHYS_HPLEV = 4.135667662E-15 # Planck's constant, error (25) in last digits [ eV s ]
PHYS_HBAR = 1.054571800E-34 # reduced Planck's constant, error (13) in last digits [ J s ]
PHYS_HBAREV = 6.582119514E-16 # reduced Planck's constant, error (40) in last digits [ eV s ]
PHYS_QEL = 1.6021766208E-19 # elementary charge, error (98) in last digits [ C ]

EL_M0 = 9.10938356E-31 # electron rest mass, error (11) in last digits [ kg ]
EL_E0 = EL_M0 * PHYS_C**2 # electron rest energy [ J ]
EL_E0EV = EL_E0 / PHYS_QEL # electron rest energy [ eV ]
EL_E0KEV = EL_E0 / PHYS_QEL / 1000. # electron rest energy [ keV ]

EL_WLKEV = PHYS_C * PHYS_HPL / PHYS_QEL * 1.0E+6 # electron wave length scale from energy [ nm keV ]

EL_CFFA = EL_M0 * PHYS_QEL**2 / (2.0 * PHYS_HPL**2) / (4. * np.pi * PHYS_EPS0) * 1.0E-10 # Coulombic form factor amplitude : m0 e^2 / ( 2 h^2 ) / ( 4 Pi eps0 ) *10^-10 [ -> A^-1 ]

def ht2wl(ht):
    '''
    Calculates the electron wavelength in nm from the kinetic energy (ht) in keV
    '''
    return EL_WLKEV / np.sqrt(ht * (ht + 2. * EL_E0KEV))

def relcor(ekv):
    '''
    Calculates the relativistic correction factor gamma for electrons of energy ekin [keV]
    '''
    return (EL_E0KEV + ekv) / EL_E0KEV

def k2theta(e0, q, de=0.):
    """

    Returns the scattering angle theta for scattering with momentum transfer hbar*q.
    An energy loss de>0 can be specified to calculate scattering angles for inelastic scattering.

    Parameters
    ----------
        e0 : float
            electron kinetic energy in eV
        q : float
            length of the vector of momentum transfer devided by hbar in 1/nm units
        de : float
            electron energy loss in eV, default: 0

    Return
    ------
        theta : float
            scattering angle in radians [0, pi]
    """
    theta = 0.
    if (np.abs(de) >= e0):
        print("Warning: energy not conserved, de is larger or equal e0.")
        return theta
    k0 = 1. / ht2wl(e0 * 0.001) # electron wave number (total momentum)
    kp = 1. / ht2wl((e0-np.abs(de)) * 0.001) # electron wave number after scattering (by conservation of energy)
    if (q < k0-kp):
        print("Warning: momentum not conserved, q is smaller than k0 - k' = {:.5E} ".format(k0-kp))
        return theta
    if (q > k0+kp):
        print("Warning: momentum not conserved, q is larger than k0 + k' = {:.5E} ".format(k0+kp))
        return np.pi
    # draw the triangle of k0 and kp enclosing theta and q opposite to theta
    # apply the law of cosine
    # q**2 = k0**2 + kp**2 - 2 * k0 * kp * cos(theta)
    # solve for cos(theta)
    costheta = (k0**2 + kp**2 - q**2) / (2 * k0 * kp)
    # invert
    theta = np.arccos(costheta)
    return theta

def kperp2theta(e0, qperp):
    """

    Returns the scattering angle theta for a given perpendicular component of the scattering vector.
    This assumes elastic forward scattering (0 <= theta < pi/2)

    Parameters
    ----------
        e0 : float
            electron kinetic energy in eV
        qperp : float
            perpendicular component of the scattering vector in 1/nm units

    Return
    ------
        theta : float
            scattering angle in radians [0, pi]
    """
    theta = 0.
    k0 = 1. / ht2wl(e0 * 0.001) # electron wave number (total momentum)
    theta = np.arcsin(qperp / k0)
    return theta