# -*- coding: utf-8 -*-
"""
Created on Mon May 03 08:46:00 2021
@author: ju-bar

Functions handling input and output of structure data
via the CEL file format.

The CEL file format is a text format defining a volume in space
which is filled by atoms.

The supercell is defined by a sequence of numbers in the second
line of the file. The line starts with a '0' characters, followed
by three numbers which set the lengths a, b, and c of the cell
edges in nanometers and three more number setting the angles
alpha, beta, and gamma  in degrees between these edges.

Atoms are defined by further lines of text, with one line for
each atom in the cell. The element is defined by symbol and the
atomic position in fractional coordinates of the cell. An occupancy
factor between 0 and 1 must be given as well as a temperature factor
using the B parameter of Debye Waller factors in nanometer square
units. Three more numbers are expected at the end of each atom line
but these numbers have no effect.

A '*' character should be placed in the last line to signalize the
end of structure input.

Example:
# This is the first line of the example defining a Si crystal in [110] orientation
  0   0.38400  0.54305  0.38400 90.00000 90.00000 90.00000
   Si  0.000000 0.000000 0.250000 1.000000 0.005400 0.000000 0.000000 0.000000
   Si  0.000000 0.250000 0.750000 1.000000 0.005400 0.000000 0.000000 0.000000
   Si  0.500000 0.750000 0.250000 1.000000 0.005400 0.000000 0.000000 0.000000
   Si  0.500000 0.500000 0.750000 1.000000 0.005400 0.000000 0.000000 0.000000
*

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import re
import numpy as np
from emilys.structure.supercell import supercell
import emilys.structure.atom as ato
import emilys.structure.atomtype as aty

def get_atom_str_CEL(a):
    """
    Returns a string of atom object parameters in CEL file format.

    Parameters
    ----------
        
        a : atom
            atom object

    Returns
    -------
        string

    """
    assert isinstance(a, ato.atom), 'This expects input of type emilys.structure.atom.atom'
    symb = aty.atom_type_symbol[a.Z]
    biso = 8. * np.pi**2 * a.uiso * 0.01 # biso in nm^2
    scrg = ato.get_str_from_charge(a.charge)
    s_out = ' {:s} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}'.format(
            symb + scrg, a.pos[0], a.pos[1], a.pos[2], a.occ, biso)
    return s_out + '  0.000000  0.000000  0.000000'

def set_atom_str_CEL(s):
    """
    Returns an atom object with parameters set from input string s.

    Parameters
    ----------
        
        s : string
            CEL file line defining an atom in a super cell
            '<symbol> <frac-x> <frac-x> <frac-z> <occ> <biso nm^2> 0 0 0'

    Returns
    -------
        atom

    """
    assert len(s) >= 11, 'insufficient length of input string'
    assert isinstance(s, str), "input should be a str object"
    l_cmp = re.split(' +|,|;|\t+', s.strip()) # decompose to list of string
    symb, in_charge = ato.get_symb_charge(l_cmp[0])
    in_Z = aty.Z_from_symb(symb) # atom type
    in_pos = np.array([float(l_cmp[1]), float(l_cmp[2]), float(l_cmp[3])])
    in_occ = float(l_cmp[4])
    in_uiso = float(l_cmp[5]) * 100. / (8. * np.pi**2) #  usio in Angst^2
    return ato.atom(Z=in_Z, pos = in_pos, uiso=in_uiso, occ=in_occ, charge=in_charge)

def read_CEL(file, debug=False):
    """
    
    Reads atomic structure data from a file assuming the CEL file format.

    Parameters
    ----------

        file : string
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
        if debug: print('dbg (read_cel): opened file [' + file + ']')
        lines = []
        for line in file_in:
            lines.append(line)
        file_in.close()
    if debug: print('dbg (read_cel): read ', len(lines),' lines of text.')
    assert len(lines) > 2, "The input file doesn't contain sufficient number of text lines."
    # create a supercell object
    sc = supercell()
    # process the cell parameters
    l_cmp = re.split(' +|,|;|\t+', lines[1].strip()) # decompose to list of string
    assert len(l_cmp) > 6, "The second line of the input file could not be split into >6 items."
    if debug: print('dbg (read_cel): supercell input line: ', l_cmp)
    sc.a0 = np.array([float(l_cmp[1]),float(l_cmp[2]),float(l_cmp[3])]) * 10. # from nm to Angst
    if debug: print('dbg (read_cel): supercell size [A]: a = {:.5f}, a = {:.5f}, c = {:.5f}'.format(*sc.a0))
    sc.angles = np.array([float(l_cmp[4]),float(l_cmp[5]),float(l_cmp[6])])
    if debug: print('dbg (read_cel): supercell angles [deg]: alpha = {:.4f}, beta = {:.4f}, gamma = {:.4f}'.format(*sc.angles))
    # process the lines
    for i in range(2, len(lines)):
        if '*' in lines[i]: break # stop reading due to end of structure character
        at_in = set_atom_str_CEL(lines[i])
        sc.l_atoms.append(at_in)
    if debug: print('dbg (read_cel): added ', len(sc.l_atoms), ' atoms to the supercell.')
    return sc

def write_CEL(sc, file):
    """

    Writes atomic structure data in CEL file format.

    Parameters
    ----------

        sc : emilys.structure.supercell.supercell
            Supercell data the should be written to file.
        file : str
            File name
    
    Returns
    -------

        int
            Error code

    """
    io_err = 0
    assert isinstance(sc, supercell), 'This expects that sc is input of type emilys.structure.supercell.supercell'
    assert isinstance(file, str), 'This expects that file is input of type str'
    with open(file, "w") as file_out:
        file_out.write("# EMILYS celio of [" + sc.get_composition_str() + "] \n")
        file_out.write(" 0  {:.5f}  {:.5f}  {:.5f}  {:.4f}  {:.4f}  {:.4f} \n".format(
            sc.a0[0], sc.a0[1], sc.a0[2], sc.angles[0], sc.angles[1], sc.angles[2]))
        for a in sc.l_atoms:
            file_out.write(get_atom_str_CEL(a) + "\n")
        file_out.write("*\n")
        file_out.close()
    return io_err