# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:18:00 2022
@author: ju-bar

EMD file I/O

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

import json
import h5py
import numpy as np

def get_data_as_str(data):
    d = np.asanyarray(data)
    string_ = [chr(i[0]) for i in d]
    return ''.join(string_).strip('\x00')

def read_emd(name, debug = False):
    """

    Reads image data from an EMD file.

    The data is returned in a dictionary with primary keys given by
    detector names. For each detector, metadata is in the secondary
    key 'Metadata' and the image data is in the secondary key 'Data'.
    A secondary key 'FrameLookupTable' is present.
    The image data is in ordering [nframe, ny, nx].

    Parameters
    ----------

        str : name
            File name

    Returns
    -------
        dict : data and metadata

    """

    d_out = {}

    # load the file
    if debug: print('emd_read_stem: reading file [' + name + ']')
    d_emd = h5py.File(name, 'r')

    d_img = d_emd['Data']['Image'] # get the /Data/Image HDF5 group
    l_img_keys = list(d_img.keys()) # get the group names (cryptic)

    if debug: print('- found', len(l_img_keys), 'image groups')

    for img_group in l_img_keys: # loop over image groups
        if debug: print('- image group = ' + img_group)
        if debug: print('  loading metadata ...')
        d = json.loads(get_data_as_str(d_img[img_group]['Metadata'])) #  get the metadata
        key_detector = 'Detector-' + d['BinaryResult']['DetectorIndex'] # get the detector index
        name_detector = d['Detectors'][key_detector]['DetectorName'] # get the detector name
        if debug: print('    detector name = ' + name_detector)

        # generate a group with name = img_group and basic items
        d_out[name_detector] = {'Metadata' : {}, 'FrameLookupTable' : [], 'Data' : []}
                
        # store metadata in the dictionary under key = img_group
        d_out[name_detector]['Metadata'] = d
        # ...   frame lookup table
        d_out[name_detector]['FrameLookupTable'] = np.asanyarray(d_img[img_group]['FrameLookupTable']).reshape(d_img[img_group]['FrameLookupTable'].shape)
        
        if debug: print('  loading image data ...')
        # ...   the image data
        dt = d_img[img_group]['Data'].dtype
        h5data = d_img[img_group]['Data']
        data = np.empty(h5data.shape)
        h5data.read_direct(data)
        data = np.rollaxis(data, axis=2) # Set the axes in frame, y, x order
        d_out[name_detector]['Data'] = data.astype(dt)
        if debug: print('    image data shape =', d_out[name_detector]['Data'].shape, '  type =', d_out[name_detector]['Data'].dtype)
        del h5data
        del data
        

    # close the file
    d_emd.close()

    return d_out