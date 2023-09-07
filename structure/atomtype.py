# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:34:00 2021
@author: ju-bar

Functions and classes handling atom type data

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

atom_type_symbol = ['Void'  ,
					'H'	, 'He'	, 'Li'	, 'Be'	, 'B'	, 'C'	, 'N'	, 'O'	, 'F'	, 'Ne'	,
			 		'Na'	, 'Mg'	, 'Al'	, 'Si'	, 'P'	, 'S'	, 'Cl'	, 'Ar'	, 'K'	, 'Ca'	,
			 		'Sc'	, 'Ti'	, 'V'	, 'Cr'	, 'Mn'	, 'Fe'	, 'Co'	, 'Ni'	, 'Cu'	, 'Zn'	,
			 		'Ga'	, 'Ge'	, 'As'	, 'Se'	, 'Br'	, 'Kr'	, 'Rb'	, 'Sr'	, 'Y'	, 'Zr'	,
			 		'Nb'	, 'Mo'	, 'Tc'	, 'Ru'	, 'Rh'	, 'Pd'	, 'Ag'	, 'Cd'	, 'In'	, 'Sn'	,
			 		'Sb'	, 'Te'	, 'I'	, 'Xe'	, 'Cs'	, 'Ba'	, 'La'	, 'Ce'	, 'Pr'	, 'Nd'	,
			 		'Pm'	, 'Sm'	, 'Eu'	, 'Gd'	, 'Tb'	, 'Dy'	, 'Ho'	, 'Er'	, 'Tm'	, 'Yb'	,
			 		'Lu'	, 'Hf'	, 'Ta'	, 'W'	, 'Re'	, 'Os'	, 'Ir'	, 'Pt'	, 'Au'	, 'Hg'	,
			 		'Tl'	, 'Pb'	, 'Bi'	, 'Po'	, 'At'	, 'Rn'	, 'Fr'	, 'Ra'	, 'Ac'	, 'Th'	,
			 		'Pa'	, 'U'	, 'Np'	, 'Pu'	, 'Am'	, 'Cm'	, 'Bk'	, 'Cf'	, 'Es'	, 'Fm'	,
			 		'Md'	, 'No'	, 'Lr'	, 'Rf'	, 'Db'	, 'Sg'	, 'Bh'	, 'Hs'	, 'Mt'	, 'Ds'	,
			 		'Rg'	, 'Cn'	, 'Nh'	, 'Fl'	, 'Mc'	, 'Lv'	, 'Ts'	, 'Og',]

def Z_from_symb(symb):
    """

    Given atomic symbol, symb, return the atomic number Z.

    Example: Z_from_symb('Ti') -> 22.

    """
    lsy = str(symb)
    lsx = lsy.translate(lsy.maketrans(dict.fromkeys('+-.0123456789','')))
    return [s.upper() for s in atom_type_symbol].index(lsx.upper())



def Z_from_str(s):
    """
    Given a string this return the atomic number Z using the first or the first two letters.
    
    Example: Z_from_str('Ti') -> 22.
    """
    Znum = -1
    s2 = s[:2].upper()
    for i in range(0, len(atom_type_symbol)):
        if s2 in atom_type_symbol[i].upper():
            Znum = i
            break
    #
    if Znum < 0:
        s1 = s[:1].upper()
        for i in range(0, len(atom_type_symbol)):
            if s1 in atom_type_symbol[i].upper():
                Znum = i
                break
    return Znum
      