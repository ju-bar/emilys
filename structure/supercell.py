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

        copy():
            Returns a copy of the supercell object.

        report(num_atoms_max):
            Prints a short report of the supercell parameters.

        get_basis():
            Calculates basis vectors from parameters a0 and angles.
            Returns a 3 x 3 numpy array where rows are basis vectors.

        get_composition_str():
            Returns a string representing the atom content of the supercell.

        keep_atoms(l_atoms_idx):
            Removes all atoms which are not indexed in list l_atoms_idx.

        delete_atoms(l_atoms_idx):
            Removes all atoms which are indexed in list l_atoms_idx.

        periodic():
            Applies periodic boundary conditions to all atoms such that
            their fractional coordinates are >=0 and <1.

        set_uiso(l_atoms_idx, uiso):
            Sets the uiso parameter of all atoms indexed by list
            l_atoms_idx to the given value.

        set_biso(l_atoms_idx, biso):
            Sets the uiso parameter of all atoms indexed by list
            l_atoms_idx by translating the given biso value.
            uiso = biso / (8 Pi**2)

        set_occ(l_atoms_idx, occ):
            Sets the occupancy parameter of all atoms indexed by list
            l_atoms_idx to the given value.

        shift_atoms(l_atoms_idx, shift, periodic):
            Shifts atoms indexed in list l_atoms_idx by a shift vector
            in fractional coordinates.

        shift_all_atoms(shift, periodic):
            Shifts all atoms by a shift vector in fractional coordinates.

        shift_atoms_to(l_atoms_idx, pos, fraction, confinement, mode):
            Shifts atoms indexed in list l_atoms_idx towards pos by
            a given fraction of the initial distance. Shifts can be confined
            to be parallel to planes or lines.

        list_positions(l_atoms_idx):
            Returns a list of positions of atoms identified by index.

        list_atoms_in_range(dic_range):
            Returns a list of atoms which parameters fall into all range
            specifications listed in the dictionary dic_range. See the
            function definition on how to setup the dictionary.

        

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
                    ', pos = [{:.5f}, {:.5f}, {:.5f}]'.format(ato.pos[0], ato.pos[1], ato.pos[2]) +
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
        """

        Returns a string reflecting the composition of the supercell.

        """
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
        assert isinstance(l_atoms_idx, list), 'This expects that parameter l_atoms_idx is a list of integers'
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
        assert isinstance(l_atoms_idx, list), 'This expects that parameter l_atoms_idx is a list of integers'
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

    def periodic(self):
        """

        Wraps all atoms periodically back to the cell so their
        fractional coordinates are >=0 and <1.

        """
        if len(self.l_atoms) > 0:
            for at in self.l_atoms:
                p = at.pos
                at.pos = np.round( p % 1.0, 6 ) # wrap with precision of 6 digits

    def set_uiso(self, l_atoms_idx, uiso):
        """

        Sets the uiso parameter of all atoms indexed by list
            l_atoms_idx to the given value.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be kept. Other atoms will be removed.

            uiso : float
                Isotropic mean square amplitude of thermal vibrations
                in Angström**2 units.

        Returns
        -------
            
            int
                Number of atoms for which the uiso value was set.

        """
        assert isinstance(l_atoms_idx, list), 'This expects that parameter l_atoms_idx is a list of integers'
        m = len(self.l_atoms) # current number of atoms
        if m == 0: return len(self.l_atoms) # nothing to delete
        n = len(l_atoms_idx) # number of atoms to delete
        l = 0
        if n > 0: # work through the list
            for i in l_atoms_idx: # copy only valid indices 
                if (i >= 0) and (i < m):
                    self.l_atoms[i].uiso = uiso
                    l += 1
        return l

    def set_biso(self, l_atoms_idx, biso):
        """

        Sets the uiso parameter of all atoms indexed by list
            l_atoms_idx by translating the given biso value.
            uiso = biso / (8 Pi**2)

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be kept. Other atoms will be removed.

            biso : float
                Isotropic B parameter of the Debye-Waller factor
                in Angström**2 units. This relates to the iostropic
                mean squared displacement amplitude usio as
                biso = 8 * Pi**2 * usio.

        Returns
        -------
            
            int
                Number of atoms for which the uiso value was set.

        """
        uiso = biso / (8. * np.pi**2) # from biso to uiso
        return self.set_uiso(l_atoms_idx, uiso)

    def set_occ(self, l_atoms_idx, occ):
        """

        Sets the occupancy parameter of all atoms indexed by list
            l_atoms_idx to the given value.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be kept. Other atoms will be removed.

            occ : float
                occupancy factor, clipped between 0 and 1

        Returns
        -------
            
            int
                Number of atoms for which the occupancy value was set.

        """
        assert isinstance(l_atoms_idx, list), 'This expects that parameter l_atoms_idx is a list of integers'
        m = len(self.l_atoms) # current number of atoms
        if m == 0: return len(self.l_atoms) # nothing to delete
        n = len(l_atoms_idx) # number of atoms to delete
        l = 0
        focc = min(1., max(0., occ))
        if n > 0: # work through the list
            for i in l_atoms_idx: # copy only valid indices 
                if (i >= 0) and (i < m):
                    self.l_atoms[i].occ = occ
                    l += 1
        return l

    def shift_atoms(self, l_atoms_idx, shift, periodic=False):
        """

        Shifts all atoms indexed in list l_atoms_idx by a shift vector
        in fractional coordinates.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be kept. Other atoms will be removed.

            shift : numpy ndarray((3),float)
                Shift vector in fractional coordinates

            periodic : boolean
                Flags that periodic wrap should be applied after shifting.

        Returns
        -------

            int
                Number of atoms for which were shifted.

        """
        assert isinstance(l_atoms_idx, list), 'This expects that parameter l_atoms_idx is a list of integers'
        m = len(self.l_atoms) # current number of atoms
        n = len(l_atoms_idx) # number of atoms to delete
        l = 0
        if (n > 0) and (m > 0): # work through the list
            for i in l_atoms_idx: # copy only valid indices 
                if (i >= 0) and (i < m):
                    p = self.l_atoms[i].pos + shift
                    if periodic:
                        self.l_atoms[i].pos = np.round(p % 1.0, 6)
                    else:
                        self.l_atoms[i].pos = p
                    l += 1
        return l

    def shift_all_atoms(self, shift, periodic=False):
        """

        Shifts all atoms by a shift vector in fractional coordinates.

        Parameters
        ----------

            shift : numpy ndarray((3),float)
                Shift vector in fractional coordinates

            periodic : boolean
                Flags that periodic wrap should be applied after shifting.

        Returns
        -------

            int
                Number of atoms for which were shifted.
        
        """
        m = len(self.l_atoms) # current number of atoms
        l = 0
        if m > 0: # work through the list
            for i in range(0, m): # copy only valid indices 
                p = self.l_atoms[i].pos + shift
                if periodic:
                    self.l_atoms[i].pos = np.round(p % 1.0, 6)
                else:
                    self.l_atoms[i].pos = p
                l += 1
        return l

    def shift_atoms_to(self, l_atoms_idx, pos, fraction=1., confinement=np.array([0.,0.,0.]), mode=3):
        """

        Shifts atoms indexed in list l_atoms_idx towards pos by
        a given fraction of the initial distance. Shifts can be confined
        to be parallel to planes or lines.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be kept. Other atoms will be removed.

            pos : numpy ndarray((3),float)
                Target position in fractional coordinates.

            fraction : float, default 1.
                Fraction of the initial distance to shift.
                Depending on parameter <mode>, the initial distance is
                1: to a plane through <pos> and with normal <confinement>
                2: to a line through <pos> and along direction <confinement>
                3: to the point <pos> without any confinements. 

            confinement : numpy ndarray((3),float), default [0.,0.,0.]
                Vector defining the orientation of a target plane (<mode> == 1)
                or a target line (<mode> == 2). If confinement is a zero vector
                a fallback to <mode> == 3 is used.

            mode : int, default 3
                Mode of confined shifts.
                1 : confinement to a direction perpendicular to a plane.
                2 : confinement to a direction perpendicular to a line.
                3 (and any other): no confinement

        Returns
        -------

            int
                Number of shifted atoms.

        """
        eps = 1.e-7 # small distance threshold
        assert isinstance(l_atoms_idx, list), 'Input <l_atoms_idx> should be a list of numbers.'
        assert isinstance(pos,np.ndarray), 'Input <pos> should be a numpy.ndarry object.'
        assert len(pos.flatten()) == 3, 'Input <pos> should contain 3 numberst.'
        assert type(fraction) is float, 'Input <fraction> should be of float type.'
        assert type(mode) is int, 'Input <mode> should be an integer number.'
        assert isinstance(confinement,np.ndarray), 'Input <confinement> should be a numpy.ndarry object.'
        assert len(confinement.flatten()) == 3, 'Input <confinement> should contain 3 numbers.'
        imode = mode
        l_conf = np.sqrt(np.dot(confinement,confinement)) # get confinement vector length
        if l_conf < eps: # zero confinement vector -> mode fallback to 3
            imode = 3
        else: # finite confinement vector
            n_conf = np.round(confinement / l_conf, 6) # confinement normal vector rounded to 6 digits
        if (imode < 1) or (imode > 3): imode = 3 # internal mode switch limited to 1, 2, or 3
        n = len(l_atoms_idx) # number of atoms to be moved
        m = len(self.l_atoms) # number of atoms in the supercell
        l = 0 # number of shifted atoms
        if (n > 0) and (m > 0): # try moving atoms
            for i in l_atoms_idx: # current atom index
                if (i < 0) or (i >= m): continue # skip invalid atom index
                ati = self.l_atoms[i] # current atom
                vec_d_pos = pos - ati.pos # get the vector from atom to pos
                if mode == 1: # ... to plane distance
                    vec_d = n_conf * np.dot(n_conf, vec_d_pos) # perpendicular vector from atom position to plane
                elif mode == 2: # ... to a line
                    vec_l = n_conf * np.dot(n_conf, vec_d_pos) # component of the distance vector parallel to the line
                    vec_d = vec_d_pos - vec_l # component of the distance vector perpendicular to the line
                else: # ... to pos
                    vec_d = vec_d_pos
                vec_shift = vec_d * fraction # shift vector
                p = np.round(ati.pos + vec_shift, 6) # shifted position
                ati.pos[:] = p[:]
                l += 1
        return l

    def merge(self, other_cell):
        """

        Merges atom list of <other_cell> to this object.

        Parameters
        ----------

            other_cell : emilys.structure.supercell.supercell
                Another supercell object to be merged into this object.

        Returns
        -------

            int
                Number of atoms merged to this object.

        Remarks
        -------

            This requires that <other_cell> has the same size and angles.

        """
        eps = 1.E-6
        assert isinstance(other_cell, supercell), 'This requires that <other_cell> is also a supercell object.'
        assert abs((self.a0[0] - other_cell.a0[0])/self.a0[0]) < eps, 'This requires equal box size (conflict with a0[0]).'
        assert abs((self.a0[1] - other_cell.a0[1])/self.a0[1]) < eps, 'This requires equal box size (conflict with a0[1]).'
        assert abs((self.a0[2] - other_cell.a0[2])/self.a0[2]) < eps, 'This requires equal box size (conflict with a0[2]).'
        assert abs((self.a0[0] - other_cell.a0[0])/self.a0[0]) < eps, 'This requires equal box angles (conflict with angles[0]).'
        assert abs((self.a0[1] - other_cell.a0[1])/self.a0[1]) < eps, 'This requires equal box angles (conflict with angles[1]).'
        assert abs((self.a0[2] - other_cell.a0[2])/self.a0[2]) < eps, 'This requires equal box angles (conflict with angles[2]).'
        m = len(other_cell.l_atoms)
        if m > 0:
            self.l_atoms.extend(other_cell.l_atoms)
        return m

    def list_positions(self, l_atoms_idx):
        """

        Returns a list of positions of atoms identified by index.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be kept. Other atoms will be removed.

        Returns
        -------

            list
                List of atom positions

        """
        assert isinstance(l_atoms_idx, list), 'Input <l_atoms_idx> should be a list of numbers.'
        l_pos = []
        n = len(l_atoms_idx) # number of atoms to be moved
        m = len(self.l_atoms) # number of atoms in the supercell
        if (m > 0) and (n > 0):
            for i in l_atoms_idx:
                if (i < 0) or (i >= m): continue # invalid index
                l_pos.append(self.l_atoms[i].pos)
        return l_pos

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
                if sel_key == 'rng_pos_a': # remove atoms out of position x range
                    for atom_idx in l_bk:
                        if (self.l_atoms[atom_idx].pos[0] < min_val) or (self.l_atoms[atom_idx].pos[0] >= max_val):
                             l_atoms_idx.remove(atom_idx)
                if sel_key == 'rng_pos_b': # remove atoms out of position y range
                    for atom_idx in l_bk:
                        if (self.l_atoms[atom_idx].pos[1] < min_val) or (self.l_atoms[atom_idx].pos[1] >= max_val):
                             l_atoms_idx.remove(atom_idx)
                if sel_key == 'rng_pos_c': # remove atoms out of position z range
                    for atom_idx in l_bk:
                        if (self.l_atoms[atom_idx].pos[2] < min_val) or (self.l_atoms[atom_idx].pos[2] >= max_val):
                             l_atoms_idx.remove(atom_idx)
        return l_atoms_idx