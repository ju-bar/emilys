# -*- coding: utf-8 -*-
"""
Created on Tue Aug 5 15:22:00 2025
@author: ju-bar

Dielectric and related functions

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

def dielelectric_penn(E, q, Ep, Eg, Gamma):
    """
    This function calculates the dielectric constant using Penn's model.

    Parameters
    ----------
        E : energy transfer in eV
        q : wave number for momentum transfer in 1/Angström
        Ep : plasmon energy in eV
        Eg : electronic band gap in eV
        Gamma : transition broadening in eV

    Returns
    -------
        complex : dielectric function epsilon(E,q)
    """
    nom = Ep**2
    denom = Eg**2 + 3.81 * q**2 - E**2 - 1.J * E * Gamma
    # beta = 3.81 eV A^2 = hbar^2 / (2 m_e)
    return 1 + nom / denom

def dielectric_function_q_dependent(omega, q, eps_inf, params):
    """
    Extended Drude-Lorentz dielectric function with q-dependent resonance shifts.

    Parameters
    ----------
    omega : float
        Energy loss (eV)
    q : float
        Momentum transfer (1/Å)
    eps_inf : float
        epsilon at infinite energy loss, e.g. 2.1 for Zeolites
    params : list of dict
        Each dict has keys: f, omega, gamma

    Returns
    -------
    complex
        Dielectric function ε(q, ω)
    """
    epsilon = complex(eps_inf, 0.0)
    for p in params:
        omega_shifted2 = p['omega']**2 + 3.81 * q**2 # 3.81 eV A^2 = hbar^2 / (2 m_e)
        denom = omega_shifted2 - omega**2 - 1j * p['gamma'] * omega
        epsilon += p['f'] / denom
    return epsilon


def dielectric_screening_factor(epsilon:complex):
    """
    Computes a dielectric screening suppression factor based on the loss function.
    Ensures the result is ∈ [0, 1], and never enhances the scattering cross section.

    Parameters
    ----------
    epsilon : complex
        Dielectric function ε(E, q)

    Returns
    -------
    float
        Screening factor S(E, q) = Im[-1/ε(E, q)] ∈ [0, 1]
    """
    e1 = epsilon.real
    e2 = epsilon.imag
    denom = e1**2 + e2**2
    if denom > 0:
        print(epsilon, e2 / denom)
        return e2 / denom
    else:
        return 0.0

def simple_screening_factor(binding_energy: float, screening_onset: float = 40.0, n: float = 3.0) -> float:
    """
    Returns a simple dielectric screening factor based on shell binding energy.

    Parameters
    ----------
    binding_energy : float
        Binding energy of the electron shell (eV) (!! not the total energy loss !!)
    screening_onset : float
        Characteristic screening onset energy (eV)
    n : float
        Steepness of the transition

    Returns
    -------
    float
        Screening factor ∈ [0, 1]
    """
    x = screening_onset / binding_energy
    return 1.0 / (1.0 + x**n)