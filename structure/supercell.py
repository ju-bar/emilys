# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:44:00 2021
@author: ju-bar

Modified on Mon Jun 20 11:50:00 2022 (ju-bar) adding additional data dict

Functions and classes handling supercells of atomic structures

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import emilys.structure.atomtype as aty
from emilys.structure.atom import atom, get_str_from_charge
from copy import deepcopy

class supercell:
    """

    class supercell

    Parameters
    ----------

        a0 : numpy.ndarray([a,b,c], dtype=float)
            cell lattice constants in Angst
        angles : numpy.ndarray([alpha, beta, gamma], dtype=float)
            cell lattice angles between [bc,ca,ab]
        basis : numpy.ndarray([[ax,ay,az],[bx,by,bz],[cx,cy,cz]], dtype=float)
            call basis vectors in Angst
        l_atoms : list of atom objects
            atoms contained in the super cell
        d_add : dictionary
            additional data, depends on I/O routines

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

        grow(add_size, center):
            Grows the supercell by given amount in Angst in three dimensions.
            This effectively adds empty space at the large coordinate ends
            of the box. The center option can be used to center the previous
            content in the new box.

        get_avg_pos(l_atoms_idx, proximity, periodic):
            Returns the average position of a list of atoms given by the
            index list l_atoms_idx. Only atoms closer than the proximity 
            parameter will be included and this may be checked under
            periodic boundary conditions.

        get_type_dict(l_atoms_idx):
            Returns a dictionary listing atomic types and sites assigned
            to these types for all atoms indexed in list l_atoms_idx.

        add_atom(Z, uiso, pos, occ, charge, faniso):
            Adds an atom to the structure with given parameters.

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

        set_faniso(l_atoms_idx, faniso):
            Sets the faniso parameter of all atoms indexed by list
            l_atoms_idx to the given value.

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

        list_close_atoms(l_atoms_idx, proximity, periodic):
            Returns a list of lists of atoms, which are closer than the
            proximity parameter in Angstroms. The periodic option switches
            the check of proximity under periodic boundary conditions.

        list_close_atoms_ref(pos, l_atoms_idx, proximity, periodic):
            Returns a list indices in l_atoms, for atoms which are closer
            to pos_ref than the proximity parameter in Angstroms.
            The periodic option switches the check of proximity under
            periodic boundary conditions.

        remove_close_atoms(l_atoms_idx, proximity):
            Returns a list of atom indices to be removed from l_atoms_idx.
            The remove is not performed, so that l_atoms_idx remains 
            unchanged by this routine.

        list_atoms_in_range(dic_range):
            Returns a list of atoms which parameters fall into all range
            specifications listed in the dictionary dic_range. See the
            function definition on how to setup the dictionary.

        dice_occupancy(l_atoms_idx, proximity, periodic):
            Randomly selects, which site realizes full occupancy from sites
            in list l_atoms_idx and replaces partial site occupation by
            full atom occupations.

        visualize_structure_2d(plane, atom_radius, figsize, invert_y):
            Plots the struture for a given plane projection using matplotlib.

        orient_box(new_c, new_b, new_box_size):
            Changes orientation such that the new c axis is new_c and 
            the new b axis is new_bin the old box coordinates. The new
            box size is set to new_box_size and the space is filled
            with atoms using periodic boundary conditions.

    """

    def __init__(self):
        self.a0 = np.array([1., 1., 1.]) # lattice constants [x, y, z]
        self.angles = np.array([90., 90., 90.]) # cell angles [alpha, beta, gamma] between [yz, zx, xy]
        self.basis = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]) # list of basis vectors x, y, z
        self.l_atoms = [] # list of atoms
        self.d_add = {} # additional data

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
        print('lattice constants [A]: a = {:.5f}, b = {:.5f}, c = {:.5f}'.format(self.a0[0],self.a0[1],self.a0[2]))
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
        assert np.abs(l_as[2]) > 0., 'cell angle gamma is invalid'
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

    def grow(self, add_size, center=False):
        """

        Grows the supercell by given amount in Angst in three dimensions.
        This effectively adds empty space at the large coordinate ends
        of the box. The center option can be used to center the previous
        content in the new box.

        Parameters
        ----------

            add_size : numpy.ndarray([x, y, z], dtype=float)
                Size added to the current box dimensions in Angst for
                each lattice dimension.

            center : boolean (default: False)
                If True, the previous content will be centered in the new
                bos. If False, the box will be extended towards the positive
                lattice axes.

        Returns
        -------

            numpy.ndarray([x, y, z], dtype=float) : new size of the box

        """
        new_a0 = self.a0 + add_size # new box size
        f_scale = self.a0 / new_a0 # scaling factor from old to new box size keeping actual positions
        n_atoms = len(self.l_atoms)
        if n_atoms > 0:
            for i in range(0, n_atoms):
                self.l_atoms[i].pos = self.l_atoms[i].pos * f_scale # scale fractional atom positions
        self.a0 = new_a0 # update bos size
        if center: # center previous content in new box?
            f_shift = 0.5 * add_size / self.a0 # fract. shift by half of the added size
            self.shift_all_atoms(f_shift) # shift without enforcing periodic boundary conditions
        return self.a0 # return new box size
        
    def tile(self, num_tile):
        """

        Adds copies of the cell num_tile times along the respective
        dimensions.

        Parameters
        ----------

            num_tile : numpy.ndarray([nx, ny, nz], dtype=int)
                Multipliers to tile the cell along each dimension.

        Returns
        -------

            numpy.ndarray([x, y, z], dtype=float) : new size of the box

        """
        ati = np.array(num_tile, dtype=float)
        ffac = np.reciprocal(ati)
        l_at_new = []
        n = len(self.l_atoms)
        for i in range(0, n):
            Z = self.l_atoms[i].Z
            pos0 = self.l_atoms[i].pos
            uiso = self.l_atoms[i].uiso
            occ = self.l_atoms[i].occ
            charge = self.l_atoms[i].charge
            faniso = self.l_atoms[i].faniso
            label = self.l_atoms[i].label
            for iz in range(0, num_tile[2]):
                for iy in range(0, num_tile[1]):
                    for ix in range(0, num_tile[0]):
                        cshift = np.array([ix, iy, iz], dtype=float)
                        pos = ffac * (pos0 + cshift)
                        l_at_new.append(atom(Z=Z, pos=pos, uiso=uiso, occ=occ, charge=charge, faniso=faniso, label=label))
        self.a0 = self.a0 * ati
        self.l_atoms = l_at_new
        return self.a0 # return new box size

    def get_avg_pos(self, l_atoms_idx, proximity, periodic):
        pos = np.array([0.,0.])
        npos = 0
        assert isinstance(l_atoms_idx, list), 'This expects that parameter l_atoms_idx is a list of integers'
        m = len(self.l_atoms) # current number of atoms
        if m == 0: return pos # dummy
        n = len(l_atoms_idx) # number of atoms to include
        if n == 0: return pos # dummy
        mb0 = self.get_basis().T # get the transformation matrix to transform from fractional to physical coordinates
        sdthr = proximity * proximity
        pos = self.l_atoms[l_atoms_idx[0]].pos.copy()
        npos = 1
        if n > 1:
            for i in range(1, n):
                idx = l_atoms_idx[i]
                apos = self.l_atoms[idx].pos.copy()
                dpos = apos - pos
                wpos = np.array([0.,0.,0.])
                if periodic:
                    for j in range(0,3):
                        if dpos[j] < -0.5:
                            wpos[j] = 1.
                        if dpos[j] >= 0.5:
                            wpos[j] = -1.
                bpos = apos + wpos
                dpos = np.dot(mb0, bpos - pos)
                if np.dot(dpos,dpos) < sdthr:
                    pos = (pos * npos + bpos) / (npos+1)
                    npos += 1
        if periodic:
            return np.round(pos % 1.0, 6)
        return np.round(pos, 6)

    def get_type_dict(self, l_atoms_idx=None, l_type_name_adds=None):
        """

            Returns a dictionary listing atomic types and sites assigned
            to these types for all atoms indexed in list l_atoms_idx.

            Parameters
            ----------
                l_atoms_idx : list
                    List of indices of the structures atom list to be included.
                    Can be None to use all atoms of the structure.
                l_type_name_adds : list
                    List of strings that determine additions made to the
                    type names:
                    'occ' : adds occupancy
                    'uiso' : adds the thermal vibration mean square amplitude
                    'ion' : adds the ionic charge

            Returns
            -------
                dict

        """
        if l_atoms_idx is None:
            aidx = list(range(0, len(self.l_atoms)))
        else:
            aidx = l_atoms_idx
        if l_type_name_adds is None:
            li_type_name_adds = []
        else:
            li_type_name_adds = l_type_name_adds
        d = {}
        n = len(aidx)
        m = len(self.l_atoms)
        if (m > 0 and n > 0):
            for i in aidx:
                if (i < m and i >= 0):
                    a = self.l_atoms[i]
                    s = a.get_type_name(li_type_name_adds)
                    if s in d.keys(): # only add site
                        d[s]['sites'].append(a.pos)
                        d[s]['id'].append(i)
                    else: # add new type
                        d[s] = { 'Z' : a.Z, 'occ' : a.occ, 'uiso' : a.uiso, 'ion' : a.charge, 'sites' : [a.pos], 'id' : [i] }
        return d
    
    def get_sites_in_slices(self, slice_planes=np.array([0.,1.]), periodic=False, l_atoms_idx=None, l_type_name_adds=None):
        """

        Returns a dictionary with types and their sites distributed into slices.
        The current structure is sliced along the c (z) direction at the given
        slice planes in fractional coordinates z/c.

        Parameters
        ----------
            slice_planes : numpy.ndarray, dtype=float
                fractional z coordinates of slice planes
            periodic : boolean, default: False
            l_atoms_idx : list, default: None
                List of indices of the structures atom list to be included.
                Can be None to use all atoms of the structure.
            l_type_name_adds : list, default: None
                List of strings that determine additions made to the
                type names:
                'occ' : adds occupancy
                'uiso' : adds the thermal vibration mean square amplitude
                'ion' : adds the ionic charge

        Returns
        -------
            dict

        """

        sp_used = np.array(slice_planes) # get sorted list of slice planes
        sp_used.sort()
        d_slc = { # generate slicing result dictionary
            'slice_planes' : sp_used
        }
        d_slc['num_slices'] = len(sp_used) - 1 # store number of slices
        #
        if d_slc['num_slices'] > 0:
            d_slc['slices'] = {} # generate slice information
            for islc in range(0, d_slc['num_slices']):
                s_slc = str(islc)
                d_slc['slices'][s_slc] = { # 
                    'range' : { # store slice range
                        'fractional' : np.array([sp_used[islc], sp_used[islc+1]]),
                        'absolute_A' : np.array([sp_used[islc] * self.a0[2], sp_used[islc+1] * self.a0[2]])
                    },
                    'thickness_A' : self.a0[2] * (sp_used[islc+1] - sp_used[islc]) # store slice thickness
                }

            # get atom types
            d_types = self.get_type_dict(l_atoms_idx=l_atoms_idx, l_type_name_adds=l_type_name_adds)
            #
            for s_type in d_types: # per type - sort into slices
                d_slc[s_type] = deepcopy(d_types[s_type]) # get type info copy
                d_slc[s_type]['slice'] = {} # add slicing
                d_ts = d_slc[s_type]['slice'] # programming shortcut
                for s_slc in d_slc['slices']:
                    d_ts[s_slc] = deepcopy(d_slc['slices'][s_slc])
                    # add lists for sites and ids
                    d_ts[s_slc]['sites'] = []
                    d_ts[s_slc]['id'] = []
                    fz_rng = d_ts[s_slc]['range']['fractional']
                    nat = len(d_slc[s_type]['id'])
                    if nat > 0:
                        for id_at in d_slc[s_type]['id']:
                            fz_pos = self.l_atoms[id_at].pos[2]
                            if periodic: fz_pos = np.round((fz_pos % 1.0),12) # catch fp round-off problems
                            if (fz_pos >= fz_rng[0] and fz_pos < fz_rng[1]):
                                d_ts[s_slc]['id'].append(id_at)
                                d_ts[s_slc]['sites'].append(self.l_atoms[id_at].pos)
        #
        return d_slc

    def add_atom(self, Z, uiso, pos, occ = 1.0, charge = 0.0, 
                 faniso = np.array([0.,0.,0.]), label = ""):
        """

        Adds an atom to the structure with given parameters.

        Parameters
        ----------
            Z : int
                atomic number
            uiso : float
                mean squared vibration amplitude in [Ang^2]
            pos : numpy.ndarray([x, y, z], dtype=float)
                fractional coordinates in a supercell
            occ : float, default 1.0
                fractional site occupation
            charge : float, default 0.0
                ionic charge in units of the elementary charge
            faniso : numpy.ndarray([f1, f2, angle], dtype=float)
                relative anisotropy parameters
            label : str
                atom label string
        
        Returns
        -------
            int
                The index at which the atom was added to the list member l_atoms.
        """
        ato = atom(Z, pos, uiso, occ, charge, faniso, label)
        self.l_atoms.append(ato)
        return len(self.l_atoms)-1

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
                to be deleted. Other atoms will be kept.

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
                for which the uiso parameter is set.

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
                for which the biso parameter is set.

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
    
    def set_faniso(self, l_atoms_idx, faniso):
        """

        Sets the relatice anisotropy parameters of all atoms indexed by list
            l_atoms_idx to the given value.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                for which the faniso parameter is set.

            faniso : np.array, shape(3,), dtype=float
                Relative anisotropy parameters [f1, f2, angle].

        Returns
        -------
            
            int
                Number of atoms for which the faniso value was set.

        """
        assert isinstance(l_atoms_idx, list), 'This expects that parameter l_atoms_idx is a list of integers'
        m = len(self.l_atoms) # current number of atoms
        if m == 0: return len(self.l_atoms) # nothing to delete
        n = len(l_atoms_idx) # number of atoms to delete
        l = 0
        if n > 0: # work through the list
            for i in l_atoms_idx: # copy only valid indices 
                if (i >= 0) and (i < m):
                    self.l_atoms[i].faniso = faniso
                    l += 1
        return l

    def set_occ(self, l_atoms_idx, occ):
        """

        Sets the occupancy parameter of all atoms indexed by list
            l_atoms_idx to the given value.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                for which the occupancy parameter is set.

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
                    self.l_atoms[i].occ = focc
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
                to be shifted.

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
                to shifted.

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
            n_conf = np.zeros_like(confinement)
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
        assert abs((self.angles[0] - other_cell.angles[0])/self.angles[0]) < eps, 'This requires equal box angles (conflict with angles[0]).'
        assert abs((self.angles[1] - other_cell.angles[1])/self.angles[1]) < eps, 'This requires equal box angles (conflict with angles[1]).'
        assert abs((self.angles[2] - other_cell.angles[2])/self.angles[2]) < eps, 'This requires equal box angles (conflict with angles[2]).'
        m = len(other_cell.l_atoms)
        if m > 0:
            self.l_atoms.extend(other_cell.l_atoms)
        return m

    def insert(self, other_cell, proximity):
        """

        Puts atoms from <other_cell> into this object. First removes all atoms of
        this object that would be closer than proximity to the new atoms.

        Parameters
        ----------

            other_cell : emilys.structure.supercell.supercell
                Another supercell object to be inserted into this object.

            proximity : float
                Minimum distance allowed of existing atoms to the new atoms.
                Any atom in the previous list that is closer than proximity
                to any of the new atoms will be removed before adding the new
                atoms.

        Returns
        -------

            int
                New number of atoms in this object.

        Remarks
        -------

            This requires that <other_cell> has the same size and angles.

        """
        eps = 1.E-6
        assert isinstance(other_cell, supercell), 'This requires that <other_cell> is also a supercell object.'
        assert abs((self.a0[0] - other_cell.a0[0])/self.a0[0]) < eps, 'This requires equal box size (conflict with a0[0]).'
        assert abs((self.a0[1] - other_cell.a0[1])/self.a0[1]) < eps, 'This requires equal box size (conflict with a0[1]).'
        assert abs((self.a0[2] - other_cell.a0[2])/self.a0[2]) < eps, 'This requires equal box size (conflict with a0[2]).'
        assert abs((self.angles[0] - other_cell.angles[0])/self.angles[0]) < eps, 'This requires equal box angles (conflict with angles[0]).'
        assert abs((self.angles[1] - other_cell.angles[1])/self.angles[1]) < eps, 'This requires equal box angles (conflict with angles[1]).'
        assert abs((self.angles[2] - other_cell.angles[2])/self.angles[2]) < eps, 'This requires equal box angles (conflict with angles[2]).'
        m = len(other_cell.l_atoms)
        r_check = proximity * 1.1 # make the check range a bit larger than the proximity
        fr_check = r_check * np.reciprocal(self.a0) # fractional check range radius
        if m > 0:
            for atj in other_cell.l_atoms:
                f0 = atj.pos - fr_check # lower check box bounds
                f1 = atj.pos - fr_check # upper check box bounds
                l_rng = self.list_atoms_in_range({'rng_pos_a' : [f0[0], f1[0]], 'rng_pos_b' : [f0[1], f1[1]], 'rng_pos_c' : [f0[2], f1[2]]})
                l_rem = self.remove_close_atoms(l_rng, proximity)
                if len(l_rem) > 0: self.delete_atoms(l_rem) # delete
            self.l_atoms.extend(other_cell.l_atoms) # extend atom list by atoms of other cell
        return len(self.l_atoms)

    def list_positions(self, l_atoms_idx):
        """

        Returns a list of positions of atoms identified by index.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                for which positions should be listed.

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

    def list_close_atoms(self, l_atoms_idx, proximity, periodic=True, debug=False):
        """

        Returns a list of lists of atoms, which are closer than the
        proximity parameter in nanometers. The periodic option switches
        the check of proximity under periodic boundary conditions.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be checked for mutual proximity. Atoms not included
                in the list will be ignored in the proximity checks.

            proximity : float
                Sets a threshold to which distance in Angstroms is
                identified as close.

            periodic : boolean, default: True
                Switches proximity checks under periodic boundary
                conditions.

            debug : boolean, default: False
                Switches extra debug text output.

        Returns
        -------

            list
                List of lists of atom indices, each sub-list is a set of atoms
                that are closer to each other than proximity
        
        """
        l_close = []
        assert isinstance(l_atoms_idx, list), 'Input <l_atoms_idx> should be a list of numbers.'
        m = len(self.l_atoms) # number of atoms in the supercell
        n = len(l_atoms_idx) # list of atom indices to check for proximity
        mb0 = self.get_basis().T # get the transformation matrix to transform from fractional to physical coordinates
        sdthr = proximity * proximity
        if (n > 1) and (m > 1): # need at least two atoms to check
            for i in range(0, n-1): # loop over atoms in list, exclusive the last
                idx = l_atoms_idx[i]
                vlp0 = self.l_atoms[idx].pos
                l_close_cur = [idx]
                for j in range(i+1, n): # loop over atoms behind i in the list
                    jdx = l_atoms_idx[j]
                    vlp1 = self.l_atoms[jdx].pos
                    if periodic: # fractional distance vector across periodic boundary conditions
                        vdlp = ((vlp1 - vlp0 + 0.5) % 1.0 ) - 0.5
                    else: # fractional distance vector, no periodic boundary
                        vdlp = vlp1 - vlp0
                    vdp = np.dot(mb0, vdlp) # distance vector in physical coordinates [nm]
                    sd = np.dot(vdp, vdp)
                    if sd <= sdthr: # squared distance check [nm**2]
                        if debug: print('#{:d} {:s} <-> #{:d} {:s}: d = {:.4f} nm'.format(
                            idx, aty.atom_type_symbol[self.l_atoms[idx].Z],
                            jdx, aty.atom_type_symbol[self.l_atoms[jdx].Z], np.sqrt(sd)))
                        l_close_cur.append(jdx) # add to current list
                # handle the current list of atoms close to atim idx
                if len(l_close_cur) > 1: # at least a pair?
                    l_close.append(l_close_cur) # append to output list
                    if debug: print('added list', l_close_cur)
        return l_close
    
    def list_neigbors(self, idx: int, periodic: bool = True, 
                      max_dist: float | None = None,
                      allowed_Z: list | None = None):
        """
        Lists neighbors to the atom with index idx sorted by distance.

        Parameters
        ----------
            idx : int
                Index of the reference atom in member l_atoms.
            periodic : bool
                (optional) Flags use of periodic boundary conditions.
            max_dist : float, default: None
                (optional) maximum distance to include in the analysis in physical length units used by the supecell
            allowed_Z : list, default: None
                (optional) Filter for allowed Z considered as neigbors

        Returns
        -------
            list, list
                List of indices and list of distances, sorted by ascending
                distance in physical length units used by the supercell.
        """
        m = len(self.l_atoms) # number of atoms in the supercell
        assert idx < m, "Reference atom index is out of range."
        mb0 = self.get_basis().T # get the transformation matrix to transform from fractional to physical coordinates
        pos_ref = self.l_atoms[idx].pos
        neighbors = [] # init neighbors list
        for jdx in range(m): # collect distances
            if jdx == idx:
                continue
            pos_test = self.l_atoms[jdx].pos
            if periodic: # fractional distance vector across periodic boundary conditions
                fdist = ((pos_test - pos_ref + 0.5) % 1.0 ) - 0.5
            else: # fractional distance vector, no periodic boundary
                fdist = pos_test - pos_ref
            vec_dist = np.dot(mb0, fdist) # distance vector in physical coordinates
            dist = np.linalg.norm(vec_dist) # real distance
            if max_dist is None or dist <= max_dist:
                if allowed_Z is None or self.l_atoms[jdx].Z in allowed_Z:
                    neighbors.append((jdx, dist))
        # Sort all neighbors by distance
        neighbors.sort(key=lambda x: x[1])
        l_atom_idx, l_dist = zip(*neighbors) if neighbors else ([], [])
        return list(l_atom_idx), list(l_dist)

    def check_duplicate_atom(self, ato, proximity=1.E-3, periodic=True):
        n = len(self.l_atoms)
        mb0 = self.get_basis().T # get the transformation matrix to transform from fractional to physical coordinates
        sdthr = proximity * proximity
        if n > 0:
            for idx in range(0, n):
                ichk = 0 # reset checker
                # distance check
                vlp = self.l_atoms[idx].pos
                if periodic: # fractional distance vector across periodic boundary conditions
                    vdlp = ((vlp - ato.pos + 0.5) % 1.0 ) - 0.5
                else: # fractional distance vector, no periodic boundary
                    vdlp = vlp - ato.pos
                vdp = np.dot(mb0, vdlp) # distance vector in physical coordinates [A]
                sd = np.dot(vdp, vdp)
                if sd <= sdthr: # squared distance check [A**2]
                    ichk += 1
                # type check
                if ato.Z == self.l_atoms[idx].Z:
                    ichk += 2
                # are there other checks to do?
                # final result
                if ichk == 3: # duplicate found
                    return True
        return False

    def list_close_atoms_ref(self, pos, l_atoms_idx, proximity, periodic=True, debug=False):
        """

        Returns a list of lists of atoms, which are closer than the
        proximity parameter in nanometers. The periodic option switches
        the check of proximity under periodic boundary conditions.

        Parameters
        ----------

            pos : numpy ndarray((3),float)
                Reference position in fractional cell coordinates. 

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be checked for mutual proximity. Atoms not included
                in the list will be ignored in the proximity checks.

            proximity : float
                Sets a threshold to which distance in Angstroms is
                identified as close.

            periodic : boolean, default: True
                Switches proximity checks under periodic boundary
                conditions.

            debug : boolean, default: False
                Switches extra debug text output.

        Returns
        -------

            list
                List of atom indices for atoms closer to pos than
                proximity
        
        """
        l_close = []
        assert isinstance(l_atoms_idx, list), 'Input <l_atoms_idx> should be a list of numbers.'
        n = len(l_atoms_idx) # list of atom indices to check for proximity
        mb0 = self.get_basis().T # get the transformation matrix to transform from fractional to physical coordinates
        sdthr = proximity * proximity
        if n > 0: # need at least one atom to check
            for i in range(0, n): # loop over atoms in list
                idx = l_atoms_idx[i]
                vlp = self.l_atoms[idx].pos
                if periodic: # fractional distance vector across periodic boundary conditions
                    vdlp = ((vlp - pos + 0.5) % 1.0 ) - 0.5
                else: # fractional distance vector, no periodic boundary
                    vdlp = vlp - pos
                vdp = np.dot(mb0, vdlp) # distance vector in physical coordinates [A]
                sd = np.dot(vdp, vdp)
                if sd <= sdthr: # squared distance check [nm**2]
                    if debug:
                        print('- #{:d} {:s}: d = {:.4f} nm'.format(
                            idx, aty.atom_type_symbol[self.l_atoms[idx].Z],np.sqrt(sd)))
                    l_close.append(idx) # add to list
        return l_close

    def remove_close_atoms(self, l_atoms_idx, proximity, debug=False):
        """

        Returns a list of atom indices to be removed from l_atoms_idx. The remove is
        not performed, so that l_atoms_idx remains unchanged by this routine.
        The list is parsed in sequence, checking proximity between atom i and atom i + x.
        Double checking should not occur in this implementation.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be checked for mutual proximity. Atoms not included
                in the list will be ignored in the proximity checks.

            proximity : float
                Sets a threshold to which distance in Angstroms is
                identified as close.

            debug : boolean, default: False
                Switches extra debug text output.

        Returns
        -------

            list
                List of atom indices that are too close to one other member of
                the input list. Removing those indices will leave a list where no
                atom is closer than proximity to any other.

        """
        l_close = []
        if debug: print('remove_close_atoms:')
        assert isinstance(l_atoms_idx, list), 'Input <l_atoms_idx> should be a list of numbers.'
        l_in = l_atoms_idx.copy() # make a copy to work with
        n = len(l_in) # list of atom indices to check for proximity
        mb0 = self.get_basis().T # get the transformation matrix to transform from fractional to physical coordinates
        sdthr = proximity * proximity
        if debug: print('- number of input items:', n)
        if (n > 1): # there are at least two atoms
            for i in range(0, n-1): # loop over atoms in list, exclusive the last
                idx = l_in[i]
                vlp0 = self.l_atoms[idx].pos
                l_rem = []
                for j in range(i+1, n): # loop over atoms behind i in the list
                    jdx = l_in[j]
                    vlp1 = self.l_atoms[jdx].pos
                    vdlp = vlp1 - vlp0 # fractional distance vector, no periodic boundary
                    vdp = np.dot(mb0, vdlp) # distance vector in physical coordinates [nm]
                    sd = np.dot(vdp, vdp) # square distance
                    if sd <= sdthr: # squared distance check [nm**2]
                        l_rem.append(j) # add local index to current removal list
                        l_close.append(jdx) # add atom index to output removal list
                # handle the current list of close atoms
                if len(l_rem) > 0: # at least one atom is too close?
                    if debug: print('- (', i, ') removing', len(l_rem), 'items ...')
                    l_rem.reverse() # reverses the local index list, so to delete from the end downwards
                    for j in l_rem: # loop local indices to remove from l_in
                        del l_in[j] # delete from l_in
                    n = len(l_in) # update lenth of l_lin
                    if debug: print('- (', i, ') remaining items:', n)
                if i >= n - 1: break # stop here, list has become too short
        return l_close

    def list_atoms_in_range(self, dic_range={}):
        """

        Returns a list of indices of atoms in member l_atoms
        whose parameters are within in all of the ranges defined
        in the dictionary dic_range.

        This is an alternative implementation

        Parameters
        ----------

            dic_range : dict, default {}
                dictionary of range definitions
                supported range keys are
                'rng_Z' : [int, int]
                    range of atomic numbers
                'lst_Z' : list of int
                    list of atomic numbers
                'rng_charge' : [float, float]
                    ionic charges
                'rng_pos_a' : [float, float]
                    fractional atom position along cell a axis
                'rng_pos_b' : [float, float]
                    fractional atom position along cell b axis
                'rng_pos_c' : [float, float]
                    fractional atom position along cell c axis
                'rng_pos_r' : [[float, float, float], float]
                    fractional atom 3d position, radius in Angs
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
        mb0 = self.get_basis().T # get the transformation matrix to transform from fractional to physical coordinates
        sdthr = 0.
        pos_ref = np.array([0., 0., 0.])
        if 'rng_pos_r' in dic_range:
            pos_ref = np.dot(mb0, dic_range['rng_pos_r'][0]) # reference position [A]
            proximity = dic_range['rng_pos_r'][1] # distance in [A]
            sdthr = proximity * proximity
        if n_atoms > 0:
            for i in range(0, n_atoms): # loop over all atoms
                ati = self.l_atoms[i]
                b_add = True # assume adding and perform logical AND operations to turn it off in case of failing conditions
                for sel_key in dic_range: # go through all conditions
                    if sel_key == 'lst_Z': # atomic number list condition
                        b_add &= (ati.Z in dic_range[sel_key])
                        continue
                    if sel_key == 'rng_pos_r': # distance in A
                        dpos = np.dot(mb0, ati.pos) - pos_ref
                        sd = np.dot(dpos, dpos) # square distance [A^2]
                        b_add &= (sd <= sdthr)
                        continue
                    min_val = min(dic_range[sel_key])
                    max_val = max(dic_range[sel_key])  
                    if sel_key == 'rng_Z': # atomic number range condition
                        b_add &= ((ati.Z >= min_val) and (ati.Z < max_val))
                        continue
                    if sel_key == 'rng_charge': # charge range
                        b_add &= ((ati.charge >= min_val) and (ati.charge < max_val))
                        continue
                    if sel_key == 'rng_uiso': # uiso range
                        b_add &= ((ati.uiso >= min_val) and (ati.uiso < max_val))
                        continue
                    if sel_key == 'rng_occ': # occupancy range
                        b_add &= ((ati.occ >= min_val) and (ati.occ < max_val))
                        continue
                    if sel_key == 'rng_pos_a': # position x range
                        b_add &= ((ati.pos[0] >= min_val) and (ati.pos[0] < max_val))
                        continue
                    if sel_key == 'rng_pos_b': # position y range
                        b_add &= ((ati.pos[1] >= min_val) and (ati.pos[1] < max_val))
                        continue
                    if sel_key == 'rng_pos_c': # position z range
                        b_add &= ((ati.pos[2] >= min_val) and (ati.pos[2] < max_val))
                        continue
                if b_add: # all conditions fulfilled ...
                    l_atoms_idx.append(i) # add atom index to list
        return l_atoms_idx

    def dice_occupancy(self, l_atoms_idx, proximity=0.01, periodic=True, debug=False):
        """

        Randomly selects, which site realizes full occupancy from sites
        in list l_atoms_idx and replaces partial site occupation by
        full atom occupations. This function modifies the list of atoms.
        This means, the input list of indices becomes invalid afterwards.

        Parameters
        ----------

            l_atoms_idx : list
                List of indices identifying atoms in member l_atoms
                to be checked for mutual proximity and occupancy selection.
                Atoms not included in the list will be ignored in the proximity
                checks.

            proximity : float, default: 0.01
                Sets a threshold to which distance in Angstroms is
                identified as close.

            periodic : boolean, default: True
                Switches proximity checks under periodic boundary
                conditions.

            debug : boolean, default: False
                Switches extra debug text output.

        Returns
        -------

            None
        
        """
        l_del = [] # list of atom indices to delete from the cell
        l_atoms_occ = [] # list of atom data to insert to l_atoms
        assert isinstance(l_atoms_idx, list), 'Input <l_atoms_idx> should be a list of numbers.'
        l_work = deepcopy(l_atoms_idx) # working list, this will get smaller as we go
        while len(l_work) > 0:
            idx = l_work[0] # get the index of the atom in l_atoms
            if self.l_atoms[idx].occ >= 1.0: # no dicing for atoms at full occupancy
                del l_work[0]
                continue
            pos_ref = self.l_atoms[idx].pos
            l_close = self.list_close_atoms_ref(pos_ref, l_work, proximity, periodic, debug)
            if len(l_close)==0: # remove this bad case, not handled
                del l_work[0]
                continue
            if debug:
                print('- working on group of {:d} sites at position'.format(len(l_close)), pos_ref)
            # setup occupancy thresholds
            l_occ_thr = []
            l_occ_total = 0.0
            for jdx in l_close:
                l_occ_total += self.l_atoms[jdx].occ
                if debug:
                    print('- #{:d}: occ={:.3f} -> total={:.3f}'.format(jdx, self.l_atoms[jdx].occ,l_occ_total))
                l_occ_thr.append(l_occ_total)
            if l_occ_total > 1.0:
                print('Warning (dice_occupancy): total occupancy >1 ({:.3f}) in proximity {:.3f} A of atom #{:d}.'.format(l_occ_total, proximity, idx))
            vrnd = np.random.rand() # random number [0,1)
            kdx = -1 # default occupation selector is None
            for j in range(0, len(l_occ_thr)): # find the occupation threshold which is bigger than vrnd
                if vrnd < l_occ_thr[j]:
                    kdx = l_close[j] # found site to fully occupy
                    break # stop looking further
            if kdx >= 0: # found something to occupy
                at_keep = deepcopy(self.l_atoms[kdx]) # this is the atom to keep
                at_keep.occ = 1.0 # fully occupy
                l_atoms_occ.append(at_keep) # store in list
                if debug:
                    print('- group of {:d} sites at position'.format(len(l_close)), 
                        pos_ref, 'occupied by ' + aty.atom_type_symbol[at_keep.Z] + 
                        ' at position', at_keep.pos)
            else:
                if debug:
                    print('- group of {:d} sites at position'.format(len(l_close)), 
                        pos_ref, 'not occupied.')
            l_del = l_del + l_close # add the current close group of sites to those to be deleted
            for jdx in l_close: # remove close group of sites from work list
                l_work.remove(jdx)
        if len(l_del) > 0: # atoms to delete from the cell
            self.delete_atoms(l_del) # deletion
        if len(l_atoms_occ) > 0: # append fully occupied atoms
            self.l_atoms = self.l_atoms + l_atoms_occ
        return
    
    def visualize_structure_2d(self, plane='xy', atom_radius=0.3, figsize=(8, 8)
                               ,show_labels=False,invert_y=False):
        """
        Visualizes a structure in 2D orthographic projection with depth-based transparency.

        Parameters:
            plane : 'xy', 'xz', or 'yz' — plane to project onto
            atom_radius : float — radius scale of atoms (in Å)
            figsize : tuple — matplotlib figure size
            show_labels : boolean - flags drawing labels
            invert_y : boolean - flags inversion of the y axis
        """
        # Extract lattice constants (assume orthorhombic)
        a, b, c = list(self.a0)

        # Choose 2D projection axes
        plane = plane.lower()
        if plane == 'xy':
            proj_indices = (0, 1)
            limits = (a, b)
        elif plane == 'xz':
            proj_indices = (0, 2)
            limits = (a, c)
        elif plane == 'yz':
            proj_indices = (1, 2)
            limits = (b, c)
        else:
            raise ValueError("Plane must be 'xy', 'xz', or 'yz'")

        # Determine the depth index (the axis orthogonal to the view)
        depth_index = list({0, 1, 2} - set(proj_indices))[0]
        depths = np.array([atom.pos[depth_index] for atom in self.l_atoms])

        # Normalize depth values to [0, 1]
        min_depth = depths.min()
        max_depth = depths.max()
        depth_range = max_depth - min_depth if max_depth != min_depth else 1.0

        # Pair atoms with their normalized depth and sort back-to-front
        sorted_atoms = sorted(
            ((atom, (atom.pos[depth_index] - min_depth) / depth_range) for atom in self.l_atoms),
            key=lambda x: x[1]  # sort by normalized depth
        )

        # Setup figure
        fig, ax = plt.subplots(figsize=figsize)

        # Color palette for atom types
        color_map = {
            'H': 'lightgray',
            'C': 'black',
            'N': 'blue',
            'O': 'red',
            'F': 'green',
            'S' : 'yellow',
            'Si': 'orange',
            'Ti': 'lightblue',
            'Ni': 'gray',
            'Sr': 'lightgreen',
            'default': 'cornflowerblue'
        }

        # Define transparency range
        min_alpha = 0.2
        max_alpha = 0.8

        # Draw atoms
        for catom, depth_norm in sorted_atoms:
            radius = 0.5 * atom_radius + catom.Z * atom_radius / 50
            pos_frac = catom.pos
            element = catom.get_type_name()
            x, y = pos_frac[proj_indices[0]], pos_frac[proj_indices[1]]

            # Convert to Cartesian
            if plane == 'xy':
                x *= a
                y *= b
            elif plane == 'xz':
                x *= a
                y *= c
            elif plane == 'yz':
                x *= b
                y *= c

            color = color_map.get(element, color_map['default'])
            alpha = min_alpha + (1 - depth_norm) * (max_alpha - min_alpha)

            circle = Circle((x, y), radius=radius,
                            facecolor=color, edgecolor='black', linewidth=0.5, alpha=alpha)
            ax.add_patch(circle)
            if show_labels:
                ax.text(x, y, catom.label, c='k', alpha=alpha+0.2, va='center', ha='center')

        ax.set_xlim(0, limits[0])
        ax.set_ylim(0, limits[1])
        ax.yaxis.set_inverted(invert_y)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel(f"{plane[0]} (Å)")
        ax.set_ylabel(f"{plane[1]} (Å)")
        ax.set_title(f"2D projection: {plane.upper()} (depth shading)")
        plt.tight_layout()
        plt.show()

    def orient_box(self, new_a, new_b, new_c):
        """
        Changes orientation using new box axes. Periodic boundary conditions are applied.

        :param object, self: Object reference
        :param array_like, new_a: New a axis in current box coordinates, e.g., [1, 1, 0]
        :param array_like, new_b: New b axis in current box coordinates, e.g., [0, 0, 1]
        :param array_like, new_c: New c axis in current box coordinates, e.g., [1, -1, 0]

        :return supercell: Returns a new supercell object.

        """
        base1 = self.get_basis() # this transforms current box coordinates to real coordinates
        ibase1 = np.linalg.inv(base1) # this transforms real coordinates to current box coordinates
        nc = supercell()
        base2t = np.array([ # new basis vectors in real coordinates
            np.dot(base1, new_a),
            np.dot(base1, new_b),
            np.dot(base1, new_c)
        ])
        print(f"new basis vectors: {base2t}")
        nc.a0 = np.array([ # new box dimensions
            np.linalg.norm(base2t[0]),
            np.linalg.norm(base2t[1]),
            np.linalg.norm(base2t[2]),
        ])
        print(f"new box dimensions: {nc.a0}")
        nc.angles = np.degrees(np.arccos(np.array([ # new box angles
            np.dot(base2t[1], base2t[2]) / (nc.a0[1]*nc.a0[2]),
            np.dot(base2t[2], base2t[0]) / (nc.a0[2]*nc.a0[0]),
            np.dot(base2t[0], base2t[1]) / (nc.a0[0]*nc.a0[1])
        ])))
        print(f"new box angles: {nc.angles}")
        base2 = nc.get_basis() # this transforms new box coordinates to real coordinates after rotation
        print(f"new basis: {base2}")
        base2 = base2t.T # new box in current real coordinates
        ibase2 = np.linalg.inv(base2) # this transforms from current real coordinates to new box coordinates
        t12 = np.matmul(ibase2, base1) # this transforms current box to new box coordinates
        print(f"box transform old->new: {t12}")
        t21 = np.linalg.inv(t12) # this transforms from new box coordinates to current box coordinates
        print(f"box transform new->old: {t21}")
        # determine tiling range of current box in new box
        # by getting the new box end-points in current box coordinates
        t_lo = np.array([0.,0.,0.], dtype=float); t_hi = np.array([0.,0.,0.], dtype=float) # initialize
        corn = [[1.,0.,0.],[1.,1.,0.],[1.,0.,1.],[0.,1.,0.],[0.,1.,1.],[0.,0.,1.],[1.,1.,1.]]
        for c in corn:
            vc1 = np.dot(t21, c) # corner of new box in current box coordinates
            for i in range(len(vc1)):
                t_lo[i] = min(t_lo[i], vc1[i])
                t_hi[i] = max(t_hi[i], vc1[i])
        it_lo = np.floor(t_lo).astype(int)
        it_hi = np.ceil(t_hi).astype(int)
        print(f"low tiling limits : {it_lo}")
        print(f"high tiling limits: {it_hi}")
        # loop over tilings and add atoms to the new supercell
        for i, ato in enumerate(self.l_atoms):
            for i2 in range(it_lo[2], it_hi[2]+1):
                for i1 in range(it_lo[1], it_hi[1]+1):
                    for i0 in range(it_lo[0], it_hi[0]+1):
                        p0 = np.array([i0, i1, i2], dtype=float)
                        p1 = p0 + ato.pos
                        p2 = np.dot(t12, p1)
                        if np.all(p2 >= 0.0) and np.all(p2 < 1.0):
                            nc.add_atom(Z=ato.Z, uiso=ato.uiso, pos=p2, 
                                        occ=ato.occ, charge=ato.charge, 
                                        faniso=ato.faniso, label=ato.label)
        return nc

