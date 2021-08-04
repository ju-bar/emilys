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

def write_XTL(sc, file, l_type_name_adds = []):
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
        file_out.write("# EMILYS xtlio of [" + sc.get_composition_str() + "] \n")
        file_out.write("  {:.6f}  {:.6f}  {:.6f}  {:.4f}  {:.4f}  {:.4f} \n".format(
            sc.a0[0], sc.a0[1], sc.a0[2], sc.angles[0], sc.angles[1], sc.angles[2]))
        d = sc.get_type_dict(None, l_type_name_adds)
        natty = len(d.keys())
        file_out.write("{:d} \n".format(natty)) # number of atom types
        for satty in d.keys():
            atty = d[satty]
            file_out.write(satty + " \n") # atom type name
            # atom type paramaters
            if ionic:
                sion = ato.get_str_from_charge(atty["ion"])
                file_out.write("{:d}  {:.3f}  {:.3f}  {:.5E}  {:s} \n".format(len(atty["sites"]),atty["Z"],atty["occ"],atty["uiso"],sion)) # number of atom types
            else:
                file_out.write("{:d}  {:.3f}  {:.3f}  {:.5E} \n".format(len(atty["sites"]),atty["Z"],atty["occ"],atty["uiso"])) # number of atom types
            # atom type sites
            for pos in atty["sites"]:
                file_out.write("     {:.6f}  {:.6f}  {:.6f} \n".format(pos[0], pos[1], pos[2]))
        file_out.write("\n")
        file_out.close()
    return io_err