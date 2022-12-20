# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:18:00 2022
@author: ju-bar

MTF file I/O

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

import numpy as np

def read_mtf(file, debug=False):
    """

    Reads radial MTF data from a text file and returns it as a 1d array.

    Parameters
    ----------

        file : str
            file name
        debug : boolean, default = False
            debug flag

    Returns
    -------

        numpy.ndarray(shape(n), dtype=float)
            radial MTF values from DC value to Nyquist
            for a detector of 2048 pixels size we expect 1025 values

    """
    assert type(file) is str, "This expects a string as input parameter."
    # open the file and read the lines
    num_lines = 0
    with open(file) as file_in:
        if debug: print('dbg (read_mtf): opened file [' + file + ']')
        lines = []
        for line in file_in:
            lines.append(line)
        file_in.close()
        num_lines = len(lines)
    if debug: print('dbg (read_mtf): read ', num_lines,' lines of text.')
    assert num_lines > 2, "The input file doesn't contain sufficient number of text lines."
    nk = int(lines[0])
    assert ((nk > 0) and (num_lines >= nk+1)), "Invalid number of radial MTF values."
    if debug: print('dbg (read_mtf): reading ', nk,' radial MTF values ...')
    a_mtf = np.zeros(nk, dtype=float)
    for ik in range(0, nk):
        a_mtf[ik] = float(lines[ik+1])
    if debug: print('dbg (read_mtf): mtf[0] =', a_mtf[0], ', mtf[Nyquist] =', a_mtf[nk-1])
    return a_mtf
