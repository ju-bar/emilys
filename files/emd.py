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

    Reads image data from an EMD file saved by the TFS Velox software.

    The data is returned in a dictionary with primary keys given by
    detector names. For each detector, metadata is in the secondary
    key 'Metadata' and the image data is in the secondary key 'Data'.
    A secondary key 'FrameLookupTable' is present.
    The image data is in ordering [nframe, ny, nx].

    Parameters
    ----------

        str : name
            File name
        boolean : debug, default : False
            Switch for debug text output

    Returns
    -------
        dict : data and metadata

    """

    d_out = {}

    # load the file
    if debug: print('emd_read_stem: reading file [' + name + ']')
    d_emd = h5py.File(name, 'r')

    main_keys = ['Version', 'Info']
    for k in main_keys:
        if k in d_emd:
            if debug: print("  reading \'" + k + "\'")
            d_out[k] = json.loads(d_emd[k][0])

    # check for image operations
    if 'Operations' in d_emd: # add operations applying to images
        if debug: print("- file contains operations information")
        d_op = d_emd['Operations']
        d_out['Operations'] = {} # create operations key in output dict
        if 'Operations' in d_op: # operation sequence ...
            if debug: print("  reading operations sequence")
            d_out['Operations']['Operations'] = json.loads(d_op['Operations'][0])
        # set target operation information to transfer also ...
        op_groups = ['CameraInputOperation', 'StemInputOperation', 
                'SurfaceReconstructionOperation', 'MathematicsOperation', 'DisplayLevelsOperation' ,
                'DpcOperation', 'IntegrationOperation', 'FftOperation']
        for k in op_groups:
            if k in d_op:
                d_op_group = d_op[k] # found a target group of operations
                if debug: print("  reading operation group " + k)
                d_out['Operations'][k] = {}
                for op in d_op_group:
                    d_out['Operations'][k][op] = json.loads(d_op_group[op][0])

    # check for image data
    d_img = d_emd['Data/Image'] # get the /Data/Image HDF5 group
    l_img_keys = list(d_img.keys()) # get the group names (cryptic)
    if debug: print('- found', len(l_img_keys), 'image groups')

    for img_group in l_img_keys: # loop over image groups
        if debug: print('- image group = ' + img_group)
        
        dataPath = '/Data/Image/' + img_group
        name_image = img_group # preset image name with the cryptic key, we are trying to replace that now by a more sensible name if possible
        # the more sensible name can be found in ...
        display_name_lookup = '/Presentation/Displays/ImageDisplay/'
        did = {}
        for key in d_emd[display_name_lookup]:
            did = json.loads(d_emd[display_name_lookup + key][0])
            if did['dataPath'] == dataPath:
                name_image = did['display']['label']
                did['operation'] = display_name_lookup + key # insert key for lookup in 'DisplayLevelsOperation'
                if debug: print('  image display name = ' + name_image + ' (used as primary key)')
                break # stop looping over display keys
        
        # generate a group with primary key name
        d_out[name_image] = {}
                
        # add primary data from the emd file for data identification
        d_out[name_image]['dataPath'] = dataPath
        d_out[name_image]['ImageDisplay'] = did

        if debug: print('  reading metadata ...')
        d = json.loads(get_data_as_str(d_img[img_group]['Metadata'])) #  get the metadata

        # store metadata in the dictionary under key = img_group
        d_out[name_image]['Metadata'] = d
        # ...   frame lookup table
        d_out[name_image]['FrameLookupTable'] = np.asanyarray(d_img[img_group]['FrameLookupTable']).reshape(d_img[img_group]['FrameLookupTable'].shape)
        
        if debug: print('  reading image data ...')
        # ...   the image data
        dt = d_img[img_group]['Data'].dtype
        h5data = d_img[img_group]['Data']
        data = np.empty(h5data.shape)
        h5data.read_direct(data)
        data = np.rollaxis(data, axis=2) # Set the axes in frame, y, x order
        d_out[name_image]['Data'] = data.astype(dt)
        if debug: print('    image data shape =', d_out[name_image]['Data'].shape, '  type =', d_out[name_image]['Data'].dtype)
        del h5data
        del data
        

    # close the file
    d_emd.close()

    return d_out