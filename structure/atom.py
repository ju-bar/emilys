# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:52:00 2021
@author: ju-bar

Functions and classes handling atom data

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import re
import numpy as np
import emilys.structure.atomtype as atty

def get_str_from_charge(charge, digits=2):
    """
    Returns a string to be attached to a symbol for a given charge.
    """
    sgn = ''
    acrg = abs(charge)
    sacrg = '{0:.{1}f}'.format(acrg,digits)
    lsa = len(sacrg)
    s1 = sacrg[:lsa-digits-1]
    s2 = sacrg[lsa-digits:]
    if int(s1)>0 or int(s2)>0:
        sgn = '+'
        if charge < 0.:
            sgn = '-'
    if int(s2) > 0:
        s2 = '.' + s2.rstrip('0')
    else:
        s2 = ''
    out = ''
    if int(s1) > 0 or len(s2) > 0:
        out = s1 + s2 + sgn
    return out

def get_symb_charge(s):
    """
    Returns symbol prefix and charge number from string s
    """
    charge = 0.
    symbol = ''
    pn = 0
    m = re.search('[0-9]+', s)
    if m:
        pn = m.start()
        symbol = s[0:pn]
        sgn = 1.
        if s[len(s)-1] == '-': sgn = -1.
        charge = float(s[pn:len(s)-1]) * sgn
    else:
        symbol = s
    return symbol, charge

def get_uiso2(uxx:float, uyy:float):
    """
    Calculates the isotropic equivalent from an in-plane anisotropic MSD.
    
    :param float, uxx: element U_xx of the tensor
    :param float, uyy: element U_yy of the tensor

    Returns: float
        uiso = (uxx + uyy)/2
    """
    return (uxx + uyy)*0.5

def get_uani_to_faniso2(uxx:float, uyy:float, uxy:float):
    """
    Calculates the fx, fy, angle parameters from an in-plane anisotropic MSD.
    
    :param uxx: element U_xx of the tensor
    :type uxx: float
    :param uyy: element U_yy of the tensor
    :type uyy: float
    :param uxy: element U_xy=U_yx of the tensor
    :type uxy: float
    
    
    Returns: (float, float, float)
        (fx, fy, angle)
    """
    ueq = get_uiso2(uxx, uyy)
    udlt = np.sqrt(0.25*(uxx-uyy)**2 + uxy**2) # tensor eigenvalue difference
    f1 = 1.0; f2 = 1.0; uang = 0. # init relative aniso params
    if np.abs(udlt) > 0.0:
        uang = np.atan2(2*uxy, uxx-uyy) # eigenvector rotation to a axis
        f1 = np.sqrt(uxx/ueq)
        f2 = np.sqrt(uyy/ueq)
    return (f1, f2, uang)

def get_faniso_to_uani2(uiso:float, f1:float, f2:float, angle:float):
    """
    Calculates the anisotropic MSD tensor from alternative parameters
    
    :param uiso: isotropic MSD
    :type uiso: float
    :param f1: axis 1 anisotropy
    :type f1: float
    :param f2: axis 2 anisotropy
    :type f2: float
    :param angle: rotation of axis 1 to x-axis
    :type angle: float

    Returns: (float, float, float)
        (uxx, uyy, uxy)
    """
    c2a = np.cos(2 * angle); s2a = np.sin(2 * angle)
    uxx = uiso * (1.0 + 0.5*(f1**2 - f2**2) * c2a)
    uyy = uiso * (1.0 - 0.5*(f1**2 - f2**2) * c2a)
    uxy = uiso * 0.5 * (f1**2 - f2**2) * s2a
    return (uxx, uyy, uxy)

class atom:
    """

    class atom

    Parameters
    ----------

        Z : integer
            atomic number
        charge : float
            ionic charge in units of the elementary charge
        uiso : float
            mean squared vibration amplitude in [Ang^2]
        occ : float
            fractional site occupation
        pos : numpy.ndarray([x, y, z], dtype=float)
            fractional coordinates in a supercell

    Methods
    -------

        get_type_name(l_type_name_adds)
            Returns a string for an atom type name with possible additions
            depending on other atom parameters.

    """

    def __init__(self, Z=1, pos=np.array([0.,0.,0.]), 
                 uiso=0.006332574, occ=1., 
                 charge=0., faniso=np.array([1.,1.,0.]),
                 label=""):
        self.Z = Z
        self.pos = pos
        self.uiso = uiso
        self.occ = occ
        self.charge = charge
        self.faniso = faniso
        self.label = label

    def get_type_name(self, l_type_name_adds = None):
        """
        
        Returns a string for an atom type name with possible additions
        depending on other atom parameters.
        
        Parameters
        ----------

            l_type_name_adds : list
                List of strings that determine additions made to the
                type names:
                'occ' : adds occupancy
                'uiso' : adds the thermal vibration mean square amplitude
                'ion' : adds the ionic charge
                'label' : label string

        """
        s = atty.atom_type_symbol[self.Z]
        if l_type_name_adds is None:
            return s
        charge = ''
        occupancy = ''
        msd = ''
        label = ''
        for addition in l_type_name_adds:
            if addition == 'ion':
                charge = get_str_from_charge(self.charge)
            if addition =='occ' :
                occupancy = f'_occ{self.occ:.3f}'
            if addition == 'uiso':
                msd = f'_uiso{self.uiso:.6f}'
            if addition == 'label':
                label = self.label
        return s + label + charge + occupancy + msd

    
