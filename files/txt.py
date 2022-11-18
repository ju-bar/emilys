# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:18:00 2022
@author: ju-bar

TTX file I/O

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

def write_table_txt(str_filename, tab, header, str_sep = '  ', str_fmt = '{:12.6E}'):
    """

    Writes a text file which contains the data of tab as
    text table.

    Parameter
    ---------
        str_filename : str
            output file name
        tab : numpy.ndarray(shape=(num_cols,num_rows))
            input data as numpy array
        header : list
            list of strings defining the header to write
        str_sep : str, default = '  '
            value separator string
        str_fmt : str, default = '{:12.6E}'
            value format command for str.format()

    """
    ndim = tab.shape
    assert len(ndim) == 2, 'Error: expecting a 2-dimensional input'
    num_cols = ndim[0]
    num_rows = ndim[1]
    file = open(str_filename, 'w+')
    tab_t = tab.T # transposed table
    if header is not None: # write headers
        if len(header) > 0: # only one header line allowed
            s_line = ''
            for sh in header:
                if len(s_line) > 0: s_line += str_sep # add separator before next header item
                s_line += sh # add next header item
            file.write(s_line + '\n')
    if num_cols > 0:
        for irow in range(0, num_rows):
            s_line = ''
            for icol in range(0, num_cols):
                if len(s_line) > 0: s_line += str_sep # add separator before next header item
                s_line += str_fmt.format(tab_t[irow,icol]) # add next header item
            file.write(s_line + '\n')
    file.close()
    
