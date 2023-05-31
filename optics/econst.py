# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 08:53:00 2020
Modified on Mon Nov 15 16:00:00 2021
@author: ju-bar

Constants and functions related to electrons and their physics

Declaration of physical constants based on published CODATA 2018
Source:	<https://doi.org/10.1103/RevModPhys.93.025010>

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
PHYS_MU0 = 1.2566370614359172953850573533118E-6 # magnetic constant = 4*Pi * 10^(-7) [ N A^(-2) ]
PHYS_EPS0 =	8.8541878176203898505365630317108E-12 # electric constant = 1/(_MU0 _C0^2) [ F m^(-1) ]
PHYS_HPL = 6.62607015E-34 # Planck's constant [ J s ]
PHYS_QEL = 1.602176634E-19 # elementary charge [ C ]

PHYS_HPLEV = 4.135667697E-15 # Planck's constant [ eV s ]
PHYS_HBAR = 1.054571818E-34 # reduced Planck's constant [ J s ]
PHYS_HBAREV = 6.582119570E-16 # reduced Planck's constant [ eV s ]
PHYS_MASSU = 1.66053906660E-27 # atomic mass constant [kg]
PHYS_KB = 1.380649E-23 # Boltzmann constant [J K^(-1)]

# Derived constants

EL_M0 = 9.1093837015E-31 # electron rest mass, error (28) in last digits [ kg ]
EL_E0 = EL_M0 * PHYS_C**2 # electron rest energy [ J ]
EL_E0EV = EL_E0 / PHYS_QEL # electron rest energy [ eV ]
EL_E0KEV = EL_E0 / PHYS_QEL / 1000. # electron rest energy [ keV ]

EL_WLKEV = PHYS_C * PHYS_HPL / PHYS_QEL * 1.0E+6 # electron wave length scale from energy [ nm keV ]

EL_CFFA = EL_M0 * PHYS_QEL**2 / (2.0 * PHYS_HPL**2) / (4. * np.pi * PHYS_EPS0) * 1.0E-10 # Coulombic form factor amplitude : m0 e^2 / ( 2 h^2 ) / ( 4 Pi eps0 ) *10^-10 [ -> A^-1 ]

def ht2wl(ht):
    '''
    Calculates the electron wavelength in nm from the kinetic energy (ht) in keV

    Parameters
    ----------
        ht : float
            beam energy in keV

    Return
    ------
        lambda : float
            electron wavelength in nm
    '''
    return EL_WLKEV / np.sqrt(ht * (ht + 2. * EL_E0KEV))

def relcor(ekv):
    '''
    Calculates the relativistic correction factor gamma for electrons of energy ekin [keV]

    Parameters
    ----------
        ht : float
            beam energy in keV

    Return
    ------
        float : rc = (m0*c^2 + ekv) / (m0*c^2)

    '''
    return (EL_E0KEV + ekv) / EL_E0KEV

def k2theta(q, e0, de=0.):
    """

    Returns the scattering angle theta for scattering with momentum transfer hbar*q.
    An energy loss de>0 can be specified to calculate scattering angles for inelastic scattering.

    Parameters
    ----------
        q : float
            length of the vector of momentum transfer devided by hbar in 1/nm units
        e0 : float
            electron kinetic energy in eV
        de : float
            electron energy loss in eV, default: 0

    Return
    ------
        theta : float
            scattering angle in radians [0, pi]
    """
    theta = 0.
    if (np.abs(de) >= e0):
        print("Warning (k2theta): energy not conserved, de is larger or equal e0.")
        return theta
    k0 = 1. / ht2wl(e0 * 0.001) # electron wave number (total momentum)
    kp = 1. / ht2wl((e0-np.abs(de)) * 0.001) # electron wave number after scattering (by conservation of energy)
    if (q < k0-kp):
        print("Warning (k2theta): momentum not conserved, q = {:.5E} is smaller than k0 - k' = {:.5E} ".format(q, k0-kp))
        return theta
    if (q > k0+kp):
        print("Warning (k2theta): momentum not conserved, q = {:.5E} is larger than k0 + k' = {:.5E} ".format(q, k0+kp))
        return np.pi
    # draw the triangle of k0 and kp enclosing theta and q opposite to theta
    # apply the law of cosine
    # q**2 = k0**2 + kp**2 - 2 * k0 * kp * cos(theta)
    # solve for cos(theta)
    costheta = (k0**2 + kp**2 - q**2) / (2 * k0 * kp)
    # invert
    theta = np.arccos(costheta)
    return theta

def kperp2theta(qperp, e0, de=0.):
    """

    Returns the scattering angle theta for a given perpendicular component of the scattering vector.
    Assumes forward scattering.

    Parameters
    ----------
        qperp : float
            perpendicular component of the scattering vector in 1/nm units
        e0 : float
            electron kinetic energy in eV
        de : float
            electron energy loss in eV, default: 0

    Return
    ------
        theta : float
            scattering angle in radians [0, pi]
    """
    theta = 0.
    if (np.abs(de) >= e0):
        print("Warning (kperp2theta): energy not conserved, de is larger or equal e0.")
        return theta
    kp = 1. / ht2wl((e0-np.abs(de)) * 0.001) # electron wave number after scattering (by conservation of energy)
    # vector of kp goes at some angle theta with respect to vector k0,
    # with qperp beeing the transversal component of vector kp.
    # thus sin(theta) = qperp / kp
    theta = np.arcsin(qperp/kp)
    return theta

def theta2qperp(theta, e0, de=0.):
    """

    Returns the perpendicular component of the scattering vector for
    a given scattering angle theta.

    Parameters
    ----------
        theta : float
            scattering angle in radians
        e0 : float
            electron kinetic energy in eV
        de : float
            electron energy loss in eV, default: 0

    Return
    ------
        qperp : float
            perpendicular component of the scattering angle in 1/nm
    """
    qperp = 0.
    if (np.abs(de) >= e0):
        print("Warning (theta2qperp): energy not conserved, de is larger or equal e0.")
        return qperp
    kp = 1. / ht2wl((e0-np.abs(de)) * 0.001) # electron wave number after scattering (by conservation of energy)
    # vector of kp goes at some angle theta with respect to vector k0,
    # with qperp beeing the transversal component of vector kp.
    # thus sin(theta) = qperp / kp
    qperp = np.sin(theta) * kp
    return qperp