# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:20:00 2024
@author: ju-bar

Functions useful for working with mean-squared displacements
in the Einstein model of independent harmonic oscillators

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
import emilys.optics.econst as ec
from scipy.optimize import fsolve

MSD_BETA = ec.PHYS_QEL / ec.PHYS_KB # thermal energy scale [K / eV]
MSD_SU0A = 0.5E20 * ec.PHYS_HBAREV**2 * ec.PHYS_QEL / ec.PHYS_MASSU # msd scale at T = 0 K [A^2 eV D]

def u2b(usqr):
    """
    Returns the B parameter used in Debye-Waller factors from
    a given mean squared displacement
    exp(-2 pi^2 <u^2> q^2) = exp(- B q^2 / 4)
    """
    return 8 * np.pi**2 * usqr

def b2u(b):
    """
    Returns the mean squared displacement for a given B parameter
    as used in the exponent in the Debye-Waller factor
    exp(-2 pi^2 <u^2> q^2) = exp(- B q^2 / 4)
    """
    return b / (8 * np.pi**2)

def u0(m, energy):
    """
    Calculates the ground state mean squared displacement
    for a given oscillator mass and vibrational energy

    Parameters
    ----------
        m : float
            oscillator mass in Dalton
        energy : float
            oscillator energy in eV

    Returns
    -------
        float
            ground state mean squared displacement in Angström^2
    """
    return MSD_SU0A / (m * energy)

def ut(m, energy, temperature):
    """
    Calculates the mean squared displacement
    for a given oscillator mass and vibrational energy
    and temperature

    Parameters
    ----------
        m : float
            oscillator mass in Dalton
        energy : float
            oscillator energy in eV
        temperature : float
            temperature in K

    Returns
    -------
        float
            mean squared displacement in Angström^2
    """
    return MSD_SU0A / (m * energy * np.tanh(0.5 * MSD_BETA * energy / temperature))

def xtx(x, a):
    """
    Thermal MSD kernel as a function of phonon energy.
    """
    return x * np.tanh(x) - a

def get_energy(usqr, m, temperature):
    """
    Returns the oscillator energy
    for a given mean squared displacement, oscillator mass
    and temperature

    Parameters
    ----------
        usqr : float
            effective mean squared displacement at given temperature
            in Angström^2
        m : float
            oscillator mass in Dalton
        temperature : float
            temperature in K

    Result
    ------
        float
            oscillator energy in eV
    """
    vt = MSD_SU0A * MSD_BETA / (2 * m * temperature * usqr)
    root = fsolve(xtx, vt, args=(vt))
    return 2 * root[0] * temperature / MSD_BETA
