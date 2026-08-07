# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 2026
@author: ju-bar

Functions handling input and output of structure data
via the PDB file format.

The PDB file format is a text format used to store biological structure
data. The format is similar to CIF. Established parsers are used to
extract the atomic structure data from the PDB file.
"""

import numpy as np
from Bio.PDB import PDBParser
from emilys.structure.atomtype import Z_from_symb
from emilys.structure.supercell import supercell

def read_PDB(filename, pad_angs=5., verbose=True):
    """
    Loads atomic structure data from a PDB file and returns
    a supercell object. Note that biological structures can
    contain large B factors depending in measurement quality.
    That means they may not represent thermal vibrations only.

    Parameters
    ----------
    filename : str
        PDB file name (filepath)
    pad_angs : float, default: 5.0
        padded space around the atoms in Angström
    verbose : bool, default: True
        Flag for producing text output

    Returns
    -------
    supercell
        an object with the atomic structure data
    """
    # --- PDB LOADING ---
    if verbose: print(f"Reading atomic structure from PDB file [{filename}] ...")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("6Z6U", filename)
    iat = 0
    xrng = [1E6, -1E6]; yrng = [1E6, -1E6]; zrng = [1E6, -1E6]
    d_elem = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Get the atom name and its x, y, z coordinates
                    elem = atom.element
                    # occ = atom.occupancy
                    coord = atom.get_coord()
                    # Update the bounding box ranges
                    xrng[0] = min(xrng[0], coord[0])
                    xrng[1] = max(xrng[1], coord[0])
                    yrng[0] = min(yrng[0], coord[1])
                    yrng[1] = max(yrng[1], coord[1])
                    zrng[0] = min(zrng[0], coord[2])
                    zrng[1] = max(zrng[1], coord[2])
                    znum = Z_from_symb(elem)
                    if znum > 0:
                        if elem not in d_elem:
                            d_elem[elem] = { "count": 0, "Z" : znum, "usio" : 1.5 / (8.0 * np.pi**2)  }
                        d_elem[elem]["count"] += 1
                    iat += 1
    if verbose:
        print(f"- number of atoms: {iat}")
        print(f"- elements: {list(d_elem.keys())}")
        num_at_el = 0
        for elem, info in d_elem.items():
            print(f"  {elem}: count = {info['count']}, Z = {info['Z']}")
            num_at_el += info['count']
        print(f"  number of atoms (from element counts): {num_at_el}")
        print(f"- box range: [{xrng}, {yrng}, {zrng}")
    # --- BOX PREP ---
    a0 = np.array([
        xrng[1] - xrng[0] + 2 * pad_angs,
        yrng[1] - yrng[0] + 2 * pad_angs,
        yrng[1] - yrng[0] + 2 * pad_angs,
        ]) # new box size
    pcent1 = np.array([
        (xrng[0] + xrng[1]) / 2, 
        (yrng[0] + yrng[1]) / 2,
        (zrng[0] + zrng[1]) / 2
        ]) # input structure center
    pcent2 = a0 * 0.5 # output structure center
    #
    sc = supercell()
    sc.a0 = a0
    sc.angles = np.array([90.0, 90.0, 90.0])
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Get the atom name and its x, y, z coordinates
                    elem = atom.element
                    occ = atom.occupancy
                    uiso = atom.bfactor / (8.0 * np.pi**2) # B -> MSD
                    coord = atom.get_coord()
                    # calculate the new coordinates after (rotation and) translation
                    new_coord = coord - pcent1 + pcent2 # np.dot(rotm, coord - pcent1) + pcent2
                    # calculate fractional coordinates
                    fpos = new_coord / sc.a0
                    sc.add_atom(d_elem[elem]['Z'], uiso, fpos, occ=occ, label=atom.get_name())
    if verbose: sc.report()
    return sc