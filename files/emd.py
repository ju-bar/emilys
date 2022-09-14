# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:18:00 2022
@author: ju-bar

EMD file I/O

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

def emd_read_ctem(name):
    """

    Reads ctem image data from an EMD file.

    Parameters
    ----------

        str : name
            File name

    Returns
    -------
        [numpy.ndarray, dict] : data, metadata

    """
    import json
    import h5py
    import numpy as np

    # load the file
    d_emd = h5py.File(name, 'r')

    d_img = d_emd['Data']['Image'] # get the /Data/Image HDF5 group
    l_img_keys = list(d_img.keys()) # get the group name (cryptic)

    # get the image as 2d array (TODO: handle multiple images)
    im0 = np.asanyarray(d_img[l_img_keys[0]]['Data']).reshape(d_img[l_img_keys[0]]['Data'].shape[0:2])
    
    # get the image metadata (TODO: combine with other file metadata)
    md0 = d_img[l_img_keys[0]]['Metadata']
    d = np.asanyarray(md0)
    string_ = [chr(i[0]) for i in d]
    string_ = ''.join(string_).strip('\x00')
    d_md0 = json.loads(string_)

    # close the file
    d_emd.close()

    return im0, d_md0
