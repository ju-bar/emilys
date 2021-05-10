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
    return [s.upper() for s in atom_type_symbol].index(symb.upper())



