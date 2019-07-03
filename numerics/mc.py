# -*- coding: utf-8 -*-
"""
Created on Sun Jul 1 12:30:00 2019
@author: ju-bar

Monte-Carlo tools

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
# %%
from numba import jit # include compilation support
import numpy as np # include numeric functions
import emilys.image.imagedata as img # include image data routines
# %%
@jit
def mc_image_pos_maximize(image, lpoints, rstep=1., etmp=0., itmax=100):
    '''

    Maximizes the intensity given by image for points in lpoints by
    moving the points in a Monte-Carlo approach.

    Returns:
        >0 = number of iterations performed.
        <0 = error code
        lpoints = points close to maximum positions

    '''
    dimg = image.shape
    dpts = lpoints.shape
    if len(dimg) != 2:
        return -10
    if len(dpts) != 2:
        return -11
    if dpts[1] != 2:
        return -12
    npts = dpts[0]
    e0 = img.image_pos_sum(image, lpoints)
    et = e0
    s = np.abs(rstep)
    it = 0
    repeat = 1
    while repeat == 1:
        ep = img.image_pos_sum(image, lpoints) # previous samples intensity
        et = ep # sampled intensity of this run
        for l in range(0,npts):
            epl = img.image_at(image, lpoints[l])
            dpos = np.random.normal(size=2) * s
            tpos = lpoints[l] + dpos
            ept = img.image_at(image, tpos)
            epd = ept - epl
            if epd > 0.: # improve
                lpoints[l] = tpos
                et = et + epd
            elif etmp > 0: # edp <= 0 maybe try
                win = epd / etmp
                pacc = 0.
                if win > -200.:
                    pacc = min(1., np.exp(win)) # finite probability to accept 0 ... 1
                if pacc > np.random.random_sample(): # accept the worse try
                    lpoints[l] = tpos
                    et = et + epd
        de = et - ep
        it = it + 1
        if it >= itmax: repeat = 0
        if de < etmp: repeat = 0 # converged
    return it