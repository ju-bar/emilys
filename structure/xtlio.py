# -*- coding: utf-8 -*-
"""
Created on Wed Aug 04 10:21:00 2021
@author: ju-bar

Functions handling input and output of structure data
via the XTL file format.

The XTL file format is a text format defining a volume in space
which is filled by atoms.

The supercell is defined by a sequence of numbers in the second
line of the file. The line starts with three numbers which set
the lengths a, b, and c of the cell edges in nanometers and
three more number setting the angles alpha, beta, and gamma
in degrees between these edges.

The third line defines the number of atom type definitions
following as sequency below.

An atom type definition consists of a variable number of text lines.
The first of these lines sets a name for the atom type, which is
can be used by the simulation program to distinguish types. This
is usually the element symbol, but can be used to distinguish
atomic sites occupied differently with the same element or for sites
of the same element but different thermal vibration. This string is
not used to determine the element type or atomic number, i.e. "R2D2"
would also be a valid input.
The second line defines properties of atomic sites for this atom type
using four numbers: (1) the number of atom sites (also the number of
lines following to define site positions), (2) the atomic number Z
(aka. core charge), (3) the occupancy factor (0.0 up to 1.0), and
(4) the mean squared displacement <u**2> in A**2 (<u**2> = B/(8 Pi**2),
where B is assumed to be an isotropic temperature factor, so that
<u**2> = (<u_x**2> + <u_y**2> + <u_z**2>)/3 with each Cartesian
component assumed to contribute equally. If ionic potentials are needed,
the charge of the ion can be added as fifth input, e.g. -1, +2, +1.3 etc.
What follows then are a numbe rof lines equal to the first number of the
preceeding line, which define atom positions by fractional coordinates
of the supercell, i.e. x/a  y/b  z/c.


Example:
# This is the first line of the example defining a BN crystal in [001] orientation
4.33704   2.50399   6.66120   90.0000   90.0000   90.0000
2
B
4    5.0000    1.000     0.0121
     0.000000  0.000000  0.250000
     0.500000  0.500000  0.250000
     0.333333  0.000000  0.750000
     0.833333  0.500000  0.750000
N
4    7.0000    1.000     0.0092
     0.000000  0.000000  0.750000
     0.500000  0.500000  0.750000
     0.333333  0.000000  0.250000
     0.833333  0.500000  0.250000


This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

import re
import numpy as np
from emilys.structure.supercell import supercell
import emilys.structure.atom as ato
import emilys.structure.atomtype as aty

def make_g_matrix(g1, g2, imax):
    a1 = np.array(g1, dtype=int) # force basis vector of integer index
    a2 = np.array(g2, dtype=int) # force basis vector of integer index
    l_g = []
    l_l = []
    for j2 in range(-imax, imax+1):
        v2 = a2 * j2
        for j1 in range(-imax, imax+1):
            v1 = a1 * j1
            v = v1 + v2
            lv = np.sqrt(np.dot(v, v))
            li = -1
            if len(l_l) > 0:
                for i in range(0, len(l_l)):
                    if l_l[i] > lv:
                        li = i
                        break
            if li >= 0:
                l_g.insert(li, v)
                l_l.insert(li, lv)
            else:
                l_g.append(v)
                l_l.append(lv)
    return l_g

def write_XTL(sc, file, l_type_name_adds = [], d_adds = {}):
    """

    Writes atomic structure data in XTL file format.

    Parameters
    ----------

        sc : emilys.structure.supercell.supercell
            Supercell data the should be written to file.
        file : str
            File name
        l_type_name_adds : list
            List of strings that determine additions made to the
            type names:
            'occ' : adds occupancy
            'uiso' : adds the thermal vibration mean square amplitude
            'ion' : adds the ionic charge, also triggers ionic charge output
        d_adds : dictionary
            Dictionary for adding additional information to the XTL file.
            Supported keys are
            'ht' : adds a high tension value (kV) in the third line
            'bragg' : adds scan and bragg vector lists, sub-list
                'sz' : zone axis vector, 3d-vector, e.g. [0, 0, 1]
                'sx' : scan vector 1, 3d-vector, e.g. [1, 0, 0]
                'sy' : scan vector 2, 3d-vector, e.g. [0, 1, 0]
                'g1' : bragg basis 1, 3d-vector, e.g. [1, 1, 0]
                    perp to sz
                'g2' : bragg basis 2, 3d-vector, e.g. [1, -1, 0]
                    perp to sz, not colinear to g1
                'gi_max' : max. bragg order to include, e.g. 5,
                    this spans a matrix by g1 and g2 up to the given number of multiples
                'gi_lim' : limits the length of the bragg vector list
    
    Returns
    -------

        int
            Error code

    """
    io_err = 0
    ionic = False
    assert isinstance(sc, supercell), 'This expects that sc is input of type emilys.structure.supercell.supercell'
    assert isinstance(file, str), 'This expects that file is input of type str'
    if 'ion' in l_type_name_adds:
        ionic = True
    with open(file, "w") as file_out:
        file_out.write(sc.get_composition_str() + " (xtlio)\n")
        file_out.write("{:<11.6f}{:<11.6f}{:<11.6f}{:<10.4f}{:<10.4f}{:<10.4f}\n".format(
            sc.a0[0], sc.a0[1], sc.a0[2], sc.angles[0], sc.angles[1], sc.angles[2]))
        if 'ht' in d_adds.keys(): # add high tension line
            file_out.write("{:.5f}\n".format(d_adds['ht']))
        d = sc.get_type_dict(None, l_type_name_adds)
        natty = len(d.keys())
        file_out.write("{:d}\n".format(natty)) # number of atom types
        for satty in d.keys():
            atty = d[satty]
            file_out.write(satty + "\n") # atom type name
            # atom type paramaters
            if ionic:
                sion = ato.get_str_from_charge(atty["ion"])
                file_out.write("{:<6d} {:<10.3f} {:<10.3f} {:<11.5E}  {:s}\n".format(len(atty["sites"]),atty["Z"],atty["occ"],atty["uiso"],sion)) # number of atom types
            else:
                file_out.write("{:<6d} {:<10.3f} {:<10.3f} {:<11.5E}\n".format(len(atty["sites"]),atty["Z"],atty["occ"],atty["uiso"])) # number of atom types
            # atom type sites
            for pos in atty["sites"]:
                file_out.write("       {:<10.6f} {:<10.6f} {:<10.6f}\n".format(pos[0], pos[1], pos[2]))
        if 'bragg' in d_adds.keys(): # add scan and bragg list
            d_b = d_adds['bragg']
            vsz = d_b['sz']
            vsx = d_b['sx']
            vsy = d_b['sy']
            gi_lim = 0
            if (('g1' in d_b.keys()) and ('g2' in d_b.keys()) and ('gi_max' in d_b.keys())):
                vg1 = d_b['g1']
                vg2 = d_b['g2']
                gi_max = d_b['gi_max']
                gi_lim = 0
                if 'gi_lim' in d_b.keys():
                    gi_lim = d_b['gi_lim']
                g_list = make_g_matrix(vg1, vg2, gi_max)
            file_out.write("{:d}\n".format(min(gi_lim, len(g_list))))
            file_out.write("   {:d}   {:d}   {:d}\n".format(vsz[0],vsz[1],vsz[2]))
            file_out.write("   {:d}   {:d}   {:d}\n".format(vsx[0],vsx[1],vsx[2]))
            file_out.write("   {:d}   {:d}   {:d}\n".format(vsy[0],vsy[1],vsy[2]))
            for i in range(0, min(gi_lim, len(g_list))):
                vg = g_list[i]
                file_out.write("   {:d}   {:d}   {:d}\n".format(vg[0],vg[1],vg[2]))
        file_out.write("\n")
        file_out.close()
    return io_err