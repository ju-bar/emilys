# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:01:00 2025
@author: ju-bar

Functions handling input of structure data
via the Vesta file format.

The Vesta file format is a text format defining a volume in space
which is filled by atoms. There is additional drawing information
for the Vesta program in the file, which will not read. Thermal
vibration parameters are not expected to be contained and need
to be set manually after reading.

No output into Vesta is available. The format is non-standard and
may change. Use CIF instead.
"""
import numpy as np
from emilys.structure.supercell import supercell
import emilys.structure.atomtype as aty

def read_vesta(filepath):
    """
    Parses CELLP and STRUC sections of a VESTA file using the multi-line format.
    
    Returns:
        supercell
            instance of class supercell with structure data
    """
    cell_params = None
    num_atoms = 0
    in_cellp = False
    in_struc = False

    with open(filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        sc = supercell()
        while i < len(lines):
            line = lines[i].strip()

            # Track sections
            if line == "CELLP":
                in_cellp = True
                in_struc = False
                i += 1
                continue
            elif line == "STRUC":
                in_struc = True
                in_cellp = False
                i += 1
                continue
            elif line.isupper() and not line.startswith("STRUC") and not line.startswith("CELLP"):
                in_cellp = False
                in_struc = False

            # Parse CELLP
            if in_cellp and cell_params is None:
                tokens = lines[i].strip().split()
                if len(tokens) == 6:
                    cell_params = list(map(float, tokens))
                    # set supercell data
                    sc.a0 = np.array([cell_params[0], cell_params[1], cell_params[2]], dtype=float)
                    sc.angles = np.array([cell_params[3], cell_params[4], cell_params[5]], dtype=float)

            # Parse STRUC (multi-line per atom)
            if in_struc and line:
                tokens = line.split()
                if len(tokens) >= 8:
                    atomic_number = aty.Z_from_symb(tokens[1])
                    occupancy = float(tokens[3])
                    x = float(tokens[4])
                    y = float(tokens[5])
                    z = float(tokens[6])
                    sc.add_atom(Z = atomic_number, uiso = 0.0, pos = np.array([x, y, z]), occ = occupancy)
                    num_atoms += 1
                    i += 1  # Skip the second line of this atom
            i += 1

    if cell_params is None:
        raise ValueError("CELLP section not found or improperly formatted.")
    if num_atoms == 0:
        raise ValueError("STRUC section not found or no atoms parsed.")

    return sc
