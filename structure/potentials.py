"""
Created on Tue Mar 21 14:35:00 2025
@author: ju-bar

Functions implementing interatomic potentials

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

Remarks: The notation is in so-called "metal" units, where
         [charge] = 1 e
         [energy] = 1 eV
         [distance] = 1 Angström

         Possible conversion to so-called "real" units is
         only required for the energy returned:
         1 eV = 23.0609 kcal / mol
         1 kcal / mol = 0.043363442 eV

"""

import numpy as np
import numba as nb


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64))
def coul(r, q1, q2, cut):
    '''
    
    Coulomb pair potential of charges q1 and q2 at distance r.
    The energy is cut off and set to zero for r > cut.

    U(r) = CE q1 q2 / r
    for 0 < r < cut

    CE = 14.399645468667815 eV A

    Parameters
    ----------
    r : float
        distance of the charges in Angström
    q1, q2 : float
        charges in elementary charge units (q=1 is a proton)
    cut : float
        cut-off distance in Angström

    Returns
    -------
    float
        potential energy in eV

    Remarks
    -------
        For r == 0.0 the function returns 0.


    '''
    if r > cut or r == 0.0:
        return 0.0
    ecoul = 14.399645468667815 * q1 * q2 / r 
    return ecoul


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
def born(r, a, rho, sigma, c, d, cut):
    '''

    Born-Mayer-Huggins or Tosi/Fumi pair potential:

    U(r) = a * exp( (sigma - r) / rho ) - c / r^6 + d / r^8
    for 0 < r < cut

    Parameters
    ----------
    r : float
        distance of the particles in Angström
    a : float
        short-range repulsion strength in eV
    rho : float
        short-range repulsion softness in Angström
    sigma : float
        combined particle radius in Angström
    c : float
        strength of attractive van der Waals term in eV A^6
    d : float
        strength of repulsive van der Waals term in eV A^8
    cut : float
        cut-off distance in Angström

    Returns
    -------
    float
        potential energy in eV

    Remarks
    -------
        For r == 0 the function returns 0.
        Fumi and Tosi, J Phys Chem Solids, 25, 31-43 (1964)

    '''
    if r > cut or r == 0.0:
        return 0.0
    ers = a * np.exp((sigma - r) / rho)
    eav = c / r**6
    erv = d / r**8
    return ers - eav + erv


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
def morse(r, d, a, r0, cut):
    '''

    Morse pair potential describing asymmetric stretching and
    compression of a covalent bond for intermolecular ions.

    U(r) = d * ( exp(-2 * a * (r - r0) ) - 2 * exp(-a * (r - r0) ) )
    for r < cut

    Parameters
    ----------
    r : float
        distance of the particles in Angström
    d : float
        depth in eV
    a : float
        steepness in 1 / Angström
    r0 : float
        equilibrium ion distance in Angström
    cut : float
        cut-off distance in Angström

    Returns
    -------
    float
        potential energy in eV

    Remarks
    -------
        P.M. Morse, Phys. Rev. 34 (1929) 57-64

    '''
    if r > cut:
        return 0.0
    ems = np.exp(-2.0 * a * (r - r0) ) # repulsive
    emc = 2 * np.exp(-1.0 * a * (r - r0) ) # attractive
    em = d * (ems - emc)
    return em