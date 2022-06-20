# -*- coding: utf-8 -*-
"""
Created on Wed Aug 04 10:21:00 2021
@author: ju-bar

Modified on Mon Jun 20 11:48:00 2022 (ju-bar) added make_g_matrix, read_XTL

Functions handling input and output of structure data
via the XTL file format.

The XTL file format is a text format defining a volume in space
which is filled by atoms.

The supercell is defined by a sequence of numbers in the second
line of the file. The line starts with three numbers which set
the lengths a, b, and c of the cell edges in nanometers and
three more number setting the angles alpha, beta, and gamma
in degrees between these edges.

The third line is the electron beam energy in keV (for most)
of the programs reading XTL, but can be (in muSTEM) the number
of atom-type definitions following as sequency below. The
decision of which option is present in the file is made on reading
line four. Line four is a number in case line three is a voltage,
then line 4 is the number of atom types. Otherwise we find an
atom type symbol in line 4. Care must be taken when checking the
content of line four, since some strings, e.g. those containing a
letter "e" or "E" could be mis-interpreted as a float number.

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
What follows then are a number of lines equal to the first number of the
preceding line, which define atom positions by fractional coordinates
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
import copy
import numpy as np
from emilys.structure.supercell import supercell
import emilys.structure.atom as ato

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

def read_XTL(file, debug=False):
    """
    
    Reads atomic structure data from a file assuming the XTL file format.

    Parameters
    ----------

        file : str
            Input file name.
        debug : boolean, default: False
            Flags debug print out of the file parsing.

    Returns
    -------

        emilys.structure.supercell.supercell

    """
    assert type(file) is str, "This expects a string as input parameter."
    # open the file and read the lines
    with open(file) as file_in:
        if debug: print('dbg (read_xtl): opened file [' + file + ']')
        lines = []
        for line in file_in:
            lines.append(line)
        file_in.close()
    if debug: print('dbg (read_xtl): read ', len(lines),' lines of text.')
    assert len(lines) > 3, "The input file doesn't contain sufficient number of text lines."
    # create a supercell object
    sc = supercell()
    # process the cell parameters
    l_cmp = re.split(' +|,|;|\t+', lines[1].strip()) # decompose to list of string
    assert len(l_cmp) > 5, "The second line of the input file could not be split into >6 items."
    if debug: print('dbg (read_xtl): supercell input line: ', l_cmp)
    sc.a0 = np.array([float(l_cmp[0]),float(l_cmp[1]),float(l_cmp[2])]) # cell size in Angst
    if debug: print('dbg (read_xtl): supercell size [A]: a = {:.5f}, a = {:.5f}, c = {:.5f}'.format(*sc.a0))
    sc.angles = np.array([float(l_cmp[3]),float(l_cmp[4]),float(l_cmp[5])])
    if debug: print('dbg (read_xtl): supercell angles [deg]: alpha = {:.4f}, beta = {:.4f}, gamma = {:.4f}'.format(*sc.angles))
    natty = 0 # preset number of atom types
    # determine option of line 3 using also line 4
    if lines[3].strip().isdigit(): # line 4 is a number
        ht = float(lines[2].strip()) # -> line 3 is the beam energy (or high tension)
        sc.d_add['ht'] = ht # write to additional data dictionary
        if debug: print('dbg (read_xtl): added beam energy [keV]: {:.4f}'.format(ht))
        natty = int(lines[3].strip()) # get number of atom type definitions following line 4
        i_line = 4 # first line of atom type data
    else: # line 4 is not a number -> line 3 is the number of atom types
        natty = int(lines[2].strip()) # get number of atom type definitions following line 3
        i_line = 3 # first line of atom type data
    if debug: print('dbg (read_xtl): number of atom types:', natty)
    
    if natty > 0 and i_line < len(lines): # there are atom types to read
        for iatty in range(0, natty):
            s_aty_label = lines[i_line].strip() # read atom type label (not used further)
            i_line += 1
            assert i_line < len(lines), "More data expected (atom type definition), file seems broken. (line #{:d})".format(i_line)
            s_aty_def = lines[i_line].strip() # read the type definition string
            i_line += 1
            l_cmp = re.split(' +|,|;|\t+',s_aty_def) # decompose to list of strings
            nat = int(l_cmp[0])
            crg = 0.0
            if len(l_cmp) > 4:
                crg = float(l_cmp[4])
            a = ato.atom(Z=int(float(l_cmp[1])), occ=float(l_cmp[2]), uiso=float(l_cmp[3]), charge=crg)
            if nat > 0: # read atom positions
                for iat in range(0, nat):
                    assert i_line < len(lines), "More data expected (atom position), file seems broken. (line #{:d})".format(i_line)
                    l_pos = re.split(' +|,|;|\t+', lines[i_line].strip())
                    i_line += 1
                    assert len(l_pos)>=3, "Three position values (fx, fy, fz), file seems broken. (line #{:d})".format(i_line)
                    f_p = [float(l_pos[0]), float(l_pos[1]), float(l_pos[2])]
                    b = ato.atom(Z=a.Z, occ=a.occ, uiso=a.uiso, charge=a.charge, pos=f_p)
                    sc.l_atoms.append(b)
    if debug: print('dbg (read_xtl): added ', len(sc.l_atoms), ' atoms to the supercell.')
    return sc