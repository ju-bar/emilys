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