# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:44:00 2022
@author: ju-bar

Functions handling input and output of structure data
via the CIF file format.

The CEL file format is a text format used here to define a volume
in space which is filled by atoms.

CIF file structure is interpreted according to the CIF dictionary
version 2.4: ftp://ftp.iucr.org/pub/cif_core.dic
See also: https://www.iucr.org/resources/cif/cif2
The main purpose is for I/O of crystal structures, additional
parameter handling will be implemented later if needed.

Example:
# CIF file example for the structure of perovskite SrTiO3 (from ICSD #80874)
_cell_length_a 3.901
_cell_length_b 3.901
_cell_length_c 3.901
_cell_angle_alpha 90.
_cell_angle_beta 90.
_cell_angle_gamma 90.
_symmetry_space_group_name_H-M 'P m -3 m'
_symmetry_Int_Tables_number 221
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 'z, y, -x'
2 'y, x, -z'
3 'x, z, -y'
4 'z, x, -y'
5 'y, z, -x'
6 'x, y, -z'
7 'z, -y, x'
8 'y, -x, z'
9 'x, -z, y'
10 'z, -x, y'
11 'y, -z, x'
12 'x, -y, z'
13 '-z, y, x'
14 '-y, x, z'
15 '-x, z, y'
16 '-z, x, y'
17 '-y, z, x'
18 '-x, y, z'
19 '-z, -y, -x'
20 '-y, -x, -z'
21 '-x, -z, -y'
22 '-z, -x, -y'
23 '-y, -z, -x'
24 '-x, -y, -z'
25 '-z, -y, x'
26 '-y, -x, z'
27 '-x, -z, y'
28 '-z, -x, y'
29 '-y, -z, x'
30 '-x, -y, z'
31 '-z, y, -x'
32 '-y, x, -z'
33 '-x, z, -y'
34 '-z, x, -y'
35 '-y, z, -x'
36 '-x, y, -z'
37 'z, -y, -x'
38 'y, -x, -z'
39 'x, -z, -y'
40 'z, -x, -y'
41 'y, -z, -x'
42 'x, -y, -z'
43 'z, y, x'
44 'y, x, z'
45 'x, z, y'
46 'z, x, y'
47 'y, z, x'
48 'x, y, z'
loop_
_atom_type_symbol
_atom_type_oxidation_number
Sr2+ 2
Ti4+ 4
O2- -2
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_symmetry_multiplicity
_atom_site_Wyckoff_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_B_iso_or_equiv
_atom_site_occupancy
_atom_site_attached_hydrogens
Sr1 Sr2+ 1 a 0 0 0 . 1. 0
Ti1 Ti4+ 1 b 0.5 0.5 0.5 . 1. 0
O1 O2- 3 c 0 0.5 0.5 . 1. 0
loop_
_atom_site_aniso_label
_atom_site_aniso_type_symbol
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_12
_atom_site_aniso_U_13
_atom_site_aniso_U_23
Sr1 Sr2+ 0.00789(3) 0.00789(3) 0.00789(3) 0 0 0
Ti1 Ti4+ 0.00558(9) 0.00558(9) 0.00558(9) 0 0 0
O1 O2- 0.00472(38) 0.0111(3) 0.0111(3) 0 0 0
#End of data

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import re
import numpy as np
from emilys.structure.supercell import supercell
import emilys.structure.atom as ato
import emilys.structure.atomtype as aty

d_cif_types = { # CIF data block type dictionary
    '_cell_length_a' : float(1.0),
    '_cell_length_b' : float(1.0),
    '_cell_length_c' : float(1.0),
    '_cell_angle_alpha' : float(1.0),
    '_cell_angle_beta' : float(1.0),
    '_cell_angle_gamma' : float(1.0),
    '_symmetry_space_group_name_H-M' : str('P 1'),
    '_symmetry_space_group_name_H-M_alt' : str('P 1'),
    '_space_group_name_H-M_alt' : str('P 1'),
    '_symmetry_Int_Tables_number' : int(1),
    '_space_group_IT_number' : int(1),
    '_symmetry_equiv_pos_site_id' : int(1),
    '_symmetry_equiv_pos_as_xyz' : str('x,y,z'),
    '_space_group_symop_id' : int(1),
    '_space_group_symop_operation_xyz' : str('x,y,z'),
    '_atom_type_symbol' : str('Fe'),
    '_atom_type_oxidation_number' : float(1.0),
    '_atom_site_label' : str('Fe'),
    '_atom_site_type_symbol' : str('Fe'),
    '_atom_site_type_charge' : float(1.0),
    '_atom_site_fract_x' : float(1.0),
    '_atom_site_fract_y' : float(1.0),
    '_atom_site_fract_z' : float(1.0),
    '_atom_site_B_iso_or_equiv' : float(1.0),
    '_atom_site_U_iso_or_equiv' : float(1.0),
    '_atom_site_occupancy' : float(1.0),
    '_atom_site_adp_type' : str('Uiso'),
    '_atom_site_thermal_displace_type' : str('Uiso'),
    '_atom_site_aniso_type_symbol' : str('Fe'),
    '_atom_site_aniso_U_11' : float(1.0),
    '_atom_site_aniso_U_22' : float(1.0),
    '_atom_site_aniso_U_33' : float(1.0),
    '_atom_site_aniso_U_12' : float(1.0),
    '_atom_site_aniso_U_13' : float(1.0),
    '_atom_site_aniso_U_23' : float(1.0),
    '_atom_site_aniso_B_11' : float(1.0),
    '_atom_site_aniso_B_22' : float(1.0),
    '_atom_site_aniso_B_33' : float(1.0),
    '_atom_site_aniso_B_12' : float(1.0),
    '_atom_site_aniso_B_13' : float(1.0),
    '_atom_site_aniso_B_23' : float(1.0)
}

d_cif_tables = {
    'symmetry_equiv_pos' : {
        'threshold' : 1.0,
        '_symmetry_equiv_pos_site_id' : 0.4,
        '_symmetry_equiv_pos_as_xyz' : 1.0,
        '_space_group_symop_id' : 0.4,
        '_space_group_symop_operation_xyz' : 1.0,
    },
    'atom_type' : {
        'threshold' : 1.0,
        '_atom_type_symbol' : 1.0,
        '_atom_type_oxidation_number' : 0.5
    },
    'atom_site' : {
        'threshold' : 4.0,
        '_atom_site_label' : 1.0,
        '_atom_site_type_symbol' : 1.0,
        '_atom_site_symmetry_multiplicity' : 0.1,
        '_atom_site_Wyckoff_symbol' : 0.1,
        '_atom_site_fract_x' : 1.0,
        '_atom_site_fract_y' : 1.0,
        '_atom_site_fract_z' : 1.0,
        '_atom_site_B_iso_or_equiv' : 1.0,
        '_atom_site_U_iso_or_equiv' : 1.0,
        '_atom_site_adp_type' : 0.1,
        '_atom_site_thermal_displace_type' : 0.1,
        '_atom_site_occupancy' : 1.0,
        '_atom_site_attached_hydrogens' : 0.1
    },
    'atom_site_aniso' : {
        'threshold' : 7.0,
        '_atom_site_aniso_label' : 1.0,
        '_atom_site_aniso_type_symbol' : 1.0,
        '_atom_site_aniso_U_11' : 1.0,
        '_atom_site_aniso_U_22' : 1.0,
        '_atom_site_aniso_U_33' : 1.0,
        '_atom_site_aniso_U_12' : 1.0,
        '_atom_site_aniso_U_13' : 1.0,
        '_atom_site_aniso_U_23' : 1.0
    }
}

str_CIF_char_qm = "'\"" # quotation marks
str_CIF_char_sep = " " # separators
str_CIF_char_com = "#" # comments
str_CIF_symop_ch = "xyz" # symmetry operation order

def get_CIF_float(s):
    val = 0.0
    rs = s.strip()
    if len(rs) > 0:
        if rs != '.' and rs != '?':
            ib = rs.find('(')
            if ib > 0: rs = rs[0:ib]
            val = float(rs)
    return val

def get_CIF_int(s):
    val = int(0)
    rs = s.strip()
    if len(rs) > 0:
        if rs != '.' and rs != '?':
            ib = rs.find('(')
            if ib > 0: rs = rs[0:ib]
            val = int(rs)
    return val

def sep_CIF_str(s):
    """

    Separates the first sub-string in s from the rest.
    This takes quotation marks into account.

    Parameters
    ----------

        s : str
            input string

    Results
    -------

        str, str
            The first substring and the remaining part.

    """
    rs = s.strip() # remove leading and trailing spaces and make copy of the input
    iqm1 = str_CIF_char_qm.find(rs[0]) # check if the first sub-string begins by one of the quotation marks
    if iqm1 >= 0: # it does ...
        iqm2 = rs[1:].find(str_CIF_char_qm[iqm1]) # .. find the closing mark
        if iqm2 >= 0: # .. found
            s1 = rs[1:(iqm2+1)] # copy the substring enclosed by quotation marks (marks removed)
            s2 = rs[(iqm2+2):].strip() # get the rest and strip
            return s1, s2 # return the two parts
    isep = rs.find(str_CIF_char_sep) # find the next separator (only one separator)
    if isep >= 0: # .. found
        s1 = rs[0:isep] # take the part in front of the separator as first sub-string
        s2 = rs[isep:].strip() # take the rest and strip
    else: # no separator, assume only one sub-string
        s1 = rs # 
        s2 = "" # empty rest
    return s1, s2

def read_CIF_scalar(s):
    """

    Reads a CIF pair of parameter name and value from the string s

    Parameters
    ----------

        s : str
            input string with a set of substrings where the second might
            be inside quotation marks

    Returns
    -------

        list : dtype = str, len = 2
            The pair of parameter name and value in form of str

    """
    l_dat = []
    rs = s.strip()
    while len(rs) > 0:
        s1, s2 = sep_CIF_str(rs)
        l_dat.append(s1)
        rs = s2
    while len(l_dat) < 2: # expand to contain n items
        l_dat.append('')
    return l_dat[0:2] # return only a list of length 2

def read_CIF_table_row(s, n):
    """

    Reads at maximum n values from the input string s

    Parameters
    ----------
        
        s : str
            input string with a set of substrings that might
            be inside quotation marks
        n : int
            number of substrings expected

    Returns
    -------

        list : dtype = str
            list of strings representing values of one table row

    """
    l_dat = []
    rs = s.strip()
    while len(rs) > 0:
        s1, s2 = sep_CIF_str(rs)
        l_dat.append(s1)
        rs = s2
    while len(l_dat) < n: # expand to contain n items
        l_dat.append('')
    return l_dat

def parse_CIF_symop_cmp(s):
    """

    Determines a vector and a scalar from the input string defining
    one line of symmetry operation in terms of a linear transformation.

    Parameters
    ----------

        s : str
            Input string, e.g. 'x' or '-x+1/2' or 'x-y'

    Returns
    -------

        np.array of shape=(4)
            The first 3 numbers are a 3-tuple defining a row of a matrix
            and the fourth number is the component of a shift vector

    """
    v = np.zeros(4, dtype=float)
    rs = s.strip().replace(' ','') # remove any whitespace character
    l = len(rs)
    if l > 2: # check for an operation -> bifurcate (recurse)
        i_op = 1 + rs[1:].find('+') # addition of two sub-terms
        if i_op > 0:
            v1 = parse_CIF_symop_cmp(rs[0:i_op])
            v2 = parse_CIF_symop_cmp(rs[i_op+1:l])
            return v1 + v2
        i_op = 1 + rs[1:].find('-') # subtraction of two sub-terms
        if i_op > 0:
            v1 = parse_CIF_symop_cmp(rs[0:i_op])
            v2 = parse_CIF_symop_cmp(rs[i_op+1:l])
            return v1 - v2
        i_op = 1 + rs[1:].find('*') # multiplication of two sub-terms
        if i_op > 0:
            v1 = parse_CIF_symop_cmp(rs[0:i_op])
            v2 = parse_CIF_symop_cmp(rs[i_op+1:l])
            # this handles cases like 2*x or 3*6
            # no second order forms, no brackets
            v[0:3] = v1[0:3] * v2[3] + v2[0:3] * v1[3]
            v[3] = v1[3] * v2[3]
            return v
        i_op = 1 + rs[1:].find('/') # multiplication of two sub-terms
        if i_op > 0:
            v1 = parse_CIF_symop_cmp(rs[0:i_op])
            v2 = parse_CIF_symop_cmp(rs[i_op+1:l])
            # this handles cases like x/2 or 1/3
            # no reciprocals, only division by numbers
            v[0:3] = v1[0:3] / v2[3]
            v[3] = v1[3] / v2[3]
            return v
    else: # trivial case, solve here (end of recurse level)
        i_ch = -1
        j = -1
        for i in range(0,3):
            j = rs.find(str_CIF_symop_ch[i])
            if j >= 0: # this channel is present
                i_ch = i # remember which character
                rs = rs[:j] + '1' + rs[(j+1):] # replace character by number
                break # stop looping
        # rs is just '1' or '-1' now
        vt = float(rs)
        if i_ch == -1:
            v[3] = vt
        else:
            v[i_ch] = vt
    return v

def get_CIF_symop(s):
    """

    Transforms a string to a linear operation consisting of
    a 3x3 matrix and a 3D-vector.

    Parameters
    ----------

        s : str
            Operation text, e.g. 'x,y,z' or '-x, y, -z + 1/2'

    Returns
    -------

        numpy.ndarray : dtype=float, shape=(4,3)
            3x3 matrix in the first 3 rows, 3D vector in the fourth row

    """
    v = np.zeros((4,3), dtype=float)
    rs = s.strip()
    l = len(rs)
    c1 = rs.find(',')
    c2 = rs.rfind(',')
    if ((c1 > 0) and (c2 < l-1) and (c2-c1 > 1)): # valid string with 2 commas
        s1 = rs[:c1]
        s2 = rs[(c1+1):(c2)]
        s3 = rs[c2+1:]
        v1 = parse_CIF_symop_cmp(s1)
        v2 = parse_CIF_symop_cmp(s2)
        v3 = parse_CIF_symop_cmp(s3)
        v[0,0:3] = v1[0:3]
        v[1,0:3] = v2[0:3]
        v[2,0:3] = v3[0:3]
        v[3,0] = v1[3]
        v[3,1] = v2[3]
        v[3,2] = v3[3]
    return v

def read_CIF(file, debug=False):
    """

    Reads a file and interprets content as atomic structure data
    given in the CIF format and returns a dictionary of the data.

    Parameters
    ----------

        file : str
            File name
        debug : boolean
            Switch debug output

    Returns
    -------

        dict
            Structure data dictionary.

    """
    d_cif = {}
    assert type(file) is str, "This expects a string as input parameter."
    # open the file and read the lines
    with open(file) as file_in:
        if debug: print('dbg (read_cif): opened file [' + file + ']')
        lines = []
        for line in file_in:
            icomm = line.find('#')
            if icomm >= 0:
                lines.append(line[0:icomm].strip()) # add lines without comment and stripped
            else:
                lines.append(line.strip()) # add lines without leading and trailing spaces
        file_in.close()
    if debug: print('dbg (read_cif): read', len(lines),'lines of text.')
    # setup book-keeper for line handling
    l_handled = np.zeros(len(lines), dtype=int)
    # find all tables "loop_"
    d_tables = {}
    for il in range(0, len(lines)): # loop over lines numerically by index
        if 0 == len(lines[il]): 
            l_handled[il] = 1
            continue # skip empty lines
        if 0 == lines[il].find('loop_'): # handle loop
            num_tab = len(d_tables.keys())
            str_tab = str(num_tab)
            d_tables[str_tab] = {
                "line_start" : il,
                "column" : [],
                "data" : []
            } # insert new table named by number
            l_handled[il] = 1
    if debug: print('dbg (read_cif): found', len(d_tables.keys()), 'tables')
    # get columns of all tables and read the table values as strings
    for str_tab in d_tables.keys():
        d_table = d_tables[str_tab]
        il = d_table['line_start'] + 1 # expect column names to start after "loop_"
        while 0 == lines[il].find('_'): # get column keys
            d_table['column'].append(lines[il]) # add column name to list
            il += 1 # next line
        num_col = len(d_table['column'])
        b_data_line = True
        while b_data_line:
            if il >= len(lines): break
            if (0 == lines[il].find('loop_')) or (0 == lines[il].find('_')) or (0 == len(lines[il])): b_data_line = False
            if b_data_line:
                d_table['data'].append(read_CIF_table_row(lines[il], num_col))
            il += 1 # next line
        num_row = len(d_table['data'])
        d_table['line_end'] = d_table['line_start'] + num_col + num_row
        l_handled[d_table['line_start']:d_table['line_end']+1] = 1
        if debug: print('dbg (read_cif): table #' + str_tab + ': #columns:', num_col, '#rows:', num_row)
    # identify table content and name it
    for str_tab in d_tables.keys():
        d_table = d_tables[str_tab]
        d_table['name'] = 'unknown' # preset unknown table data
        l_str_cif_tables = list(d_cif_tables.keys())
        l_vote = np.zeros(len(l_str_cif_tables), dtype=float) # votes from cif_tables
        for icol in range(0, len(d_table['column'])): # loop over columns of the loaded table
            str_col = d_table['column'][icol] # get column name
            for i_cif_table in range(0, len(l_str_cif_tables)): # loop over known CIF table names
                str_cif_table = l_str_cif_tables[i_cif_table]
                if str_col in d_cif_tables[str_cif_table].keys(): # check if the currently checked CIF table contains the column name
                    l_vote[i_cif_table] += d_cif_tables[str_cif_table][str_col] # ... yes, increment vote by name weight
        for i_cif_table in range(0, len(l_str_cif_tables)): # loop over votes
            str_cif_table = l_str_cif_tables[i_cif_table] # get name
            if l_vote[i_cif_table] >= d_cif_tables[str_cif_table]['threshold']: # vote exceeds threshold
                d_table['name'] = str_cif_table # set table name
                if debug: print('dbg (read_cif): table #' + str_tab + ' identified as ' + str_cif_table)
                break # do not look for a second table
    # add tables to the cif dictionary
    d_cif['tables'] = d_tables
    if debug: print('dbg (read_cif):', sum(l_handled), 'of', len(lines), 'lines handled after table reading.')
    # read scalar parameters from unhandled lines
    for il in range(0, len(lines)):
        if l_handled[il] == 0: # unhandled line
            str_prm_name, str_prm_val = read_CIF_scalar(lines[il])
            if str_prm_name in d_cif_types:
                d_cif[str_prm_name] = str_prm_val
            l_handled[il] = 1
    if debug: print('dbg (read_cif):', sum(l_handled), 'of', len(lines), 'lines handled after scalar parameter reading.')
    return d_cif

def write_CIF_qm(s):
    """
    """
    rs = str(s)
    if rs.find(' ') >= 0: # contains spaces?
        if rs.find("'") >= 0: # contains single qm?
            return '"' + rs + '"' # put in souble quotes
        else:
            return "'" + rs + "'" # put in single quotes
    return rs

def write_CIF(d_cif, file, debug=False):
    """

    Writes a CIF file from dictionary information.

    Parameters
    ----------

        d_cif : dict
            Dictionary with CIF information
        file : str
            File name
        debug : boolean, default=False
            Switch debug print

    Returns
    -------

        int
            Error code.

    """
    io_err = 0
    assert isinstance(file, str), 'This expects that file is input of type str'
    with open(file, "w") as file_out:
        file_out.write("# (cifio)\n")
        for str_CIF_key in d_cif:
            if str_CIF_key == 'tables': continue # skip tables
            file_out.write(str_CIF_key + " " + write_CIF_qm(d_cif[str_CIF_key]) + "\n")
        file_out.write("\n")
        if 'tables' in d_cif:
            for str_CIF_tab in d_cif['tables']:
                d_table = d_cif['tables'][str_CIF_tab]
                if d_table['name'] == 'unknown':
                    file_out.write("# table\n")
                else:
                    file_out.write("# table " + str(d_table['name']) + "\n")
                file_out.write("loop_\n")
                for str_col in d_table['column']:
                    file_out.write(str_col + "\n")
                for l_row in d_table['data']:
                    ll = list(l_row)
                    for i in range(0, len(ll)):
                        ll[i] = write_CIF_qm(ll[i])
                    file_out.write(' '.join(ll) + "\n")
                file_out.write("\n")
        file_out.write("#End_(cifio)\n")
    return io_err

def get_CIF_atom_sites(d_cif):
    l_as = []
    if 'tables' in d_cif:
        d_as = {}
        d_at = {}
        d_aniso = {}
        for str_tab in d_cif['tables']:
            if d_cif['tables'][str_tab]['name'] == 'atom_site':
                d_as = d_cif['tables'][str_tab]
            if d_cif['tables'][str_tab]['name'] == 'atom_type':
                d_at = d_cif['tables'][str_tab]
            if d_cif['tables'][str_tab]['name'] == 'atom_site_aniso':
                d_aniso = d_cif['tables'][str_tab]
        if ('data' in d_as) and ('column' in d_as):
            l_col = list(d_as['column'])
            for asite in d_as['data']:
                a = ato.atom()
                str_key = '_atom_site_type_symbol'
                i_key = l_col.index(str_key)
                if i_key < 0:
                    print('Error (get_CIF_atom_sites): Missing key [' + str_key + '] in table of atom sites.')
                    break
                str_symb = str(asite[i_key])
                a.Z = aty.Z_from_symb(str_symb)
                str_key = '_atom_site_type_occupancy'
                i_key = l_col.index(str_key)
                if i_key >= 0:
                    a.occ = get_CIF_float(asite[i_key])
                else:
                    a.occ = float(1.0)
                # read usio
                a.uiso = float(0.) # initialize with uiso not set
                # - check for specific adp type definition
                str_adp_type = "unknown"
                str_key = '_atom_site_adp_type'
                i_key = l_col.index(str_key)
                if i_key >= 0:
                    str_adp_type = str(asite[i_key])
                str_key = '_atom_site_thermal_displace_type'
                i_key = l_col.index(str_key)
                if i_key >= 0:
                    str_adp_type = str(asite[i_key])
                # - check for Biso column
                str_key = '_atom_site_B_iso_or_equiv'
                i_key = l_col.index(str_key)
                if i_key >= 0:
                    val = get_CIF_float(asite[i_key])
                    if str_adp_type == "unknown":
                        str_adp_type = "Biso"
                    if str_adp_type == "Uiso":
                        a.uiso = val
                    elif str_adp_type == "Biso":
                        a.uiso = val / (8. * np.pi**2) # translate from Biso input to Uiso stored
                # - check for Uiso column
                str_key = '_atom_site_U_iso_or_equiv'
                i_key = l_col.index(str_key)
                if i_key >= 0:
                    if str_adp_type == "unknown":
                        str_adp_type = "Uiso"
                    val = get_CIF_float(asite[i_key]) # uiso input
                    if str_adp_type == "Uiso":
                        a.uiso = val
                    elif str_adp_type == "Biso":
                        a.uiso = val / (8. * np.pi**2) # translate from Biso input to Uiso stored
                # - check anisotropic table and override if available for current atom type
                
                # read charge (oxidation state if possible)
                if ('data' in d_at) and ('column' in d_at):
                    l_col_type = list(d_at['column'])
                    for atype in d_at['data']:
                        str_key_type = '_atom_type_symbol'
                        i_key_type = l_col_type.index(str_key_type)
                        str_symb_type = str(atype[i_key_type])
                        if str_symb_type == str_symb: # found type match?
                            str_key_type = '_atom_type_oxidation_number'
                            i_key_type = l_col_type.index(str_key_type)
                            if i_key_type >= 0:
                                a.charge = get_CIF_float(atype[i_key_type])
                            else:
                                a.charge = float(0.)
                            break
                else:
                    a.charge = float(0.)
        else: # problem, no sites
            print('Error (get_CIF_atom_sites): No table of atom sites.')
    return l_as

def get_CIF_atom_sites_P1(d_cif):
    l_as_P1 = []
    l_as = get_CIF_atom_sites(d_cif) # get list of atom sites using atom objects
    return l_as_P1

def CIF_to_supercell(d_cif):
    sc = supercell()
    sc.a0[0] = float(d_cif['_cell_length_a'])
    sc.a0[1] = float(d_cif['_cell_length_b'])
    sc.a0[2] = float(d_cif['_cell_length_c'])
    sc.angles[0] = float(d_cif['_cell_angle_alpha'])
    sc.angles[1] = float(d_cif['_cell_angle_beta'])
    sc.angles[2] = float(d_cif['_cell_angle_gamma'])
    sc.basis = sc.get_basis()
    sc.l_atoms = get_CIF_atom_sites_P1(d_cif) # get the atom site list in space group P1
    return sc