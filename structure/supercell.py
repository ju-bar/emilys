# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:44:00 2021
@author: ju-bar

Functions and classes handling supercells of atomic structures

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

import os, sys
import numpy as np
import emilys.structure.atomtype as aty
from emilys.structure.atom import atom, get_str_from_charge
from copy import deepcopy

class supercell:
    """

    class supercell

    Parameters
    ----------

        a0 : numpy.ndarray([a, b, c], dtype=float)
            cell lattice constants
        angles : numpy.ndarray([alpha, beta, gamma], dtype=float)
            cell lattice angles between [bc, ca, ab]
        basis : numpy.ndarray([[ax, ay, az],[bx,by,bz],[cx,cy,cz]], dtype=float)
            call basis vectors
        l_atoms : list of atom objects
            atoms contained in the super cell

    Methods
    -------

        get_basis: calculates basis vectors from parameters a0 and angles

    """

    def __init__(self):
        self.a0 = np.array([1., 1., 1.]) # lattice constants [x, y, z]
        self.angles = np.array([90., 90., 90.]) # cell angles [alpha, beta, gamma] between [yz, zx, xy]
        self.basis = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]) # list of basis vectors x, y, z
        self.l_atoms = [] # list of atoms

    def copy(self):
        return deepcopy(self)

    def report(self, num_atoms_max=10):
        """

        Prints a report of the supercell parameters.

        Parameters
        ----------

            num_atoms_max : int, default 10
                Maximum number of contained atoms to be printed.

        """
        n = len(self.l_atoms)
        print('lattice constants [nm]: a = {:.5f}, b = {:.5f}, c = {:.5f}'.format(self.a0[0],self.a0[1],self.a0[2]))
        print('lattice angles [deg]: alpha = {:.4f}, beta = {:.4f}, gamma = {:.4f}'.format(self.angles[0],self.angles[1],self.angles[2]))
        print('number of atoms: {:d}'.format(n))
        n_max = min(n, num_atoms_max)
        if n_max:
            for i in range(0, n_max):
                ato = self.l_atoms[i]
                symb = aty.atom_type_symbol[ato.Z] + get_str_from_charge(ato.charge)
                print('#{:d}: '.format(i) + symb +
                    ', pos = [{:.5f}, {:.5f}, {}]'.format(ato.pos[0], ato.pos[1], ato.pos[2]) +
                    ', occ = {:.5f}, uiso = {:.5f}'.format(ato.occ, ato.uiso) )

    def get_basis(self):
        """

        Returns the list of three 3D basis vectors defining the lattice.

        """
        l_ac = np.cos( np.deg2rad(self.angles) ).round(15)
        l_as = np.sin( np.deg2rad(self.angles) ).round(15)
        assert np.abs(l_as[2]) > 0., 'cell angle gamma is inavlid'
        return np.array([
                [self.a0[0], self.a0[1] * l_ac[2], self.a0[2] * l_ac[1]],
                [0. , self.a0[1] * l_as[2], self.a0[2] * ( l_ac[0] - l_ac[1] * l_ac[2] ) / l_as[2] ],
                [0. , 0. , self.a0[2] * np.sqrt( l_as[2]**2 - l_ac[1]**2 - l_ac[0]**2 + 2. * l_ac[0] * l_ac[1] * l_ac[2] ) / l_as[2] ]
            ])
    
    def get_composition_str(self):
        s_cmp = ''
        d_cmp = {}
        for ato in self.l_atoms:
            symb = aty.atom_type_symbol[ato.Z] + get_str_from_charge(ato.charge)
            if symb in d_cmp:
                d_cmp[symb] += ato.occ
            else:
                d_cmp[symb] = ato.occ
        n_cmp = 0
        for comp in d_cmp:
            if n_cmp > 0: s_cmp += ' ' 
            s_cmp += comp + '_'
            v_occ = d_cmp[comp]
            if v_occ > int(v_occ):
                s_cmp += '{:.2f}'.format(v_occ)
            else:
                s_cmp += str(int(v_occ))
            n_cmp += 1
        return s_cmp

    def keep_atoms(self, l_atoms_idx):
        """

        Keeps atoms in the supercell that are listed by index in the
        parameter l_atoms_idx. This modifies the list of atoms of the
        supercell object.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be kept. Other atoms will be removed.

        Returns
        -------

            int
                Number of remaining atoms in the supercell object

        """
        assert type(l_atoms_idx) is list, 'This expects that parameter l_atoms_idx is a list of integers'
        m = len(self.l_atoms) # current number of atoms
        if m == 0: return 0 # nothing to keep
        n = len(l_atoms_idx) # new number of atoms
        if n > 0: # check the list
            l_work = [] # internal list
            for i in l_atoms_idx: # copy only valid indices 
                if (i >= 0) and (i < m):
                    l_work.append(i)
            n = len(l_work)
            l_work = sorted(l_work) # get index list sorted
        #
        if n == 0: # nothing to keep
            self.l_atoms.clear() # erase all
        else:
            l_atoms_cp = deepcopy(self.l_atoms) # get a copy of the atoms list
            self.l_atoms.clear() # clear the current list of atoms
            for i in l_work: # copy atoms back
                self.l_atoms.append(l_atoms_cp[i])
        return len(self.l_atoms)

    def delete_atoms(self, l_atoms_idx):
        """

        Deletes atoms from the supercell that are listed by index in the
        parameter l_atoms_idx. This modifies the list of atoms of the
        supercell object.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be kept. Other atoms will be removed.

        Returns
        -------

            int
                Number of remaining atoms in the supercell object

        """
        assert type(l_atoms_idx) is list, 'This expects that parameter l_atoms_idx is a list of integers'
        m = len(self.l_atoms) # current number of atoms
        if m == 0: return len(self.l_atoms) # nothing to delete
        n = len(l_atoms_idx) # number of atoms to delete
        if n > 0: # check the list
            l_work = [] # internal list
            for i in l_atoms_idx: # copy only valid indices 
                if (i >= 0) and (i < m):
                    l_work.append(i)
            n = len(l_work)
            l_work = sorted(l_work, reverse=True) # get index list sorted, reverse
        #
        if n == 0: return len(self.l_atoms) # nothing to delete
        for i in l_work: # copy atoms back
            del self.l_atoms[i]
        return len(self.l_atoms)

    def list_atoms_in_range(self, dic_range={}):
        """

        Returns a list of indices of atoms in member l_atoms
        whose parameters are within in all of the ranges defined
        in the dictionary dic_range.

        Parameters
        ----------

            dic_range : dict, default {}
                dictionary of range definitions
                supported range keys are
                'rng_Z' : [int, int]
                    atomic numbers
                'rng_charge' : [float, float]
                    ionic charges
                'rng_pos_a' : [float, float]
                    fractional atom position along cell a axis
                'rng_pos_b' : [float, float]
                    fractional atom position along cell b axis
                'rng_pos_c' : [float, float]
                    fractional atom position along cell c axis
                'rng_uiso' : [float, float]
                    thermal vibration amplitudes
                'rng_occ' : [float, float]
                    occupancy factors

        Returns
        -------

            list of int

        Notes
        -----

            Ranges defined will be checked inclusive of the lower
            bound and exclusive for the upper bound,
            i.e. (x0 <= x) and (x < x1).

        """
        l_atoms_idx = [] # initialize empty list of selected atom indices
        n_atoms = len(self.l_atoms) # get number of atoms in the structure
        if n_atoms > 0:
            l_atoms_idx = list(range(0, n_atoms)) # initialize with all atoms listed, corresponds to no conditions case
            for sel_key in dic_range: # go through all conditions
                l_bk = l_atoms_idx.copy()
                min_val = min(dic_range[sel_key])
                max_val = max(dic_range[sel_key])  
                if sel_key == 'rng_Z': # remove all atoms not fulfilling atomic number range condition
                    for atom_idx in l_bk:
                        if (self.l_atoms[atom_idx].Z < min_val) or (self.l_atoms[atom_idx].Z >= max_val):
                             l_atoms_idx.remove(atom_idx)
                if sel_key == 'rng_charge': # remove atoms out of charge range
                    for atom_idx in l_bk:
                        if (self.l_atoms[atom_idx].charge < min_val) or (self.l_atoms[atom_idx].charge >= max_val):
                             l_atoms_idx.remove(atom_idx)
                if sel_key == 'rng_uiso': # remove atoms out of uiso range
                    for atom_idx in l_bk:
                        if (self.l_atoms[atom_idx].uiso < min_val) or (self.l_atoms[atom_idx].uiso >= max_val):
                             l_atoms_idx.remove(atom_idx)
                if sel_key == 'rng_occ': # remove atoms out of occupancy range
                    for atom_idx in l_bk:
                        if (self.l_atoms[atom_idx].occ < min_val) or (self.l_atoms[atom_idx].occ >= max_val):
                             l_atoms_idx.remove(atom_idx)
                if sel_key == 'rng_pos_x': # remove atoms out of position x range
                    for atom_idx in l_bk:
                        if (self.l_atoms[atom_idx].pos[0] < min_val) or (self.l_atoms[atom_idx].pos[0] >= max_val):
                             l_atoms_idx.remove(atom_idx)
                if sel_key == 'rng_pos_y': # remove atoms out of position y range
                    for atom_idx in l_bk:
                        if (self.l_atoms[atom_idx].pos[1] < min_val) or (self.l_atoms[atom_idx].pos[1] >= max_val):
                             l_atoms_idx.remove(atom_idx)
                if sel_key == 'rng_pos_z': # remove atoms out of position y range
                    for atom_idx in l_bk:
                        if (self.l_atoms[atom_idx].pos[2] < min_val) or (self.l_atoms[atom_idx].pos[2] >= max_val):
                             l_atoms_idx.remove(atom_idx)
        return l_atoms_idx