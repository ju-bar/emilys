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
        if charge < 0.: sgn = '-'
    if int(s2) > 0:
        s2 = '.' + s2.rstrip('0')
    else:
        s2 = ''
    sout = ''
    if int(s1) > 0 or len(s2) > 0:
        sout = s1 + s2 + sgn
    return sout

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

    def __init__(self, Z=1, pos=np.array([0.,0.,0.]), uiso=0.006332574, occ=1., charge=0., faniso=np.array([0.,0.,0.])):
        self.Z = Z
        self.pos = pos
        self.uiso = uiso
        self.occ = occ
        self.charge = charge
        self.faniso = faniso

    def get_type_name(self, l_type_name_adds = []):
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

        """
        s = atty.atom_type_symbol[self.Z]
        scrg = ''
        socc = ''
        suiso = ''
        for sadd in l_type_name_adds:
            if sadd == 'ion':
                scrg = get_str_from_charge(self.charge) 
            if sadd =='occ' :
                socc = '_occ{:.3f}'.format(self.occ)
            if sadd == 'uiso':
                suiso = '_uiso{:.6f}'.format(self.uiso)
        return s + scrg + socc + suiso

    
