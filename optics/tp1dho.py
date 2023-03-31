# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:37:00 2023
@author: ju-bar

Functions related to the 1d quantum harmonic oscillator (QHO)

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import emilys.optics.econst as ec
from scipy.special import hermite
import numpy as np

def usqr0(m, w):
    """

    Calculates the mean squared displacement of the 1d QHO
    ground state from mass m and frequency w.
    usqr0 = hbar / (2 m w)

    parameters
    ----------

        m : float
            mass in kg
        w : float
            oscillator frequency f/2pi in 1/s
        
    returns
    -------

        float
            mean squared displacement in m^2

    """
    return ec.PHYS_HBAR / (2. * m * w)

def En(n, w):
    return (n + 0.5) * ec.PHYS_HBAR * w

def usqrn(n, m, w):
    return (2*n + 1) * usqr0(m, w)

def xrng(n, us, num_scale=100):
    num = (n + 1) * num_scale
    xmax = np.sqrt(us * (2*n + 1)) * (3. + 5./(n+1))
    dx = 2*xmax/num
    return np.arange(-xmax, xmax, dx)

def dwf(q, us):
    return np.exp(-2 * np.pi**2 * us * q**2)

def psi(x, n, us):
    """

    1d quantum harmonic oscillator wavefunction
    psi(x) = (2**n * n!)**(-1/2) * (2 * np.pi * us)**(-1/4)
             * np.exp(-x**2 / (4 * us)) 
             * hermite(n, x / (2 * us)**(1/2))

    parameters
    ----------

    x : float
        space coordinate
    n : int
        state quantum number
    us : float
        mean squared displacement of the ground state
        this relates to the oscillator frequency omega and mass m
        by us = hbar / (2 m omega), hbar = ec.PHYS_HBAR

    returns
    -------
    float : psi
        value of the wave function (this is a real-valued function)

    """
    hn = hermite(n)
    yhn = hn(x / np.sqrt(2 * us))
    yef = np.exp(-x**2 / (4 * us))
    ydf2 = (2 * np.pi * us)**0.25
    p2 = np.power(2., n)
    nf = float(np.math.factorial(n))
    ydf1 = np.sqrt(p2 * nf)
    return yhn * yef / (ydf1 * ydf2)
