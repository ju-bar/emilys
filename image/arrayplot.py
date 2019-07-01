# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:42:00 2019
@author: ju-bar
"""
# %%
import matplotlib.pyplot as plt # include mathematic plotting
import numpy as np # include numeric functions
# %%
def get_value_range(array, vrange=np.array([0.,1.]), vrangetype='default'):
    vmin = np.amin(array)
    vmax = np.amax(array)
    rmin = vmin * (1. - vrange[0])
    rmax = vmax * vrange[1]
    arrshow = array
    if vrangetype == 'zerosym': # set value range symmetric around zero
        if vmax < 0:
            vmax = np.abs(vmin)
        else:
            vmin = -vmax
        rmin = vmin * (1. - vrange[0])
        rmax = vmax * vrange[1]
    elif vrangetype == 'direct': # set the value range directly from vrange
        rmin = vrange[0]
        rmax = vrange[1]
    elif vrangetype == 'modulo': # use modulo on the value range
        rmin = vrange[0]
        rmax = vrange[1]
        arrshow = np.mod(array - vrange[0], vrange[1] -vrange[0]) + vrange[0]
    return [rmin, rmax, arrshow]
# %%
def arrayplot2d(array, pixscale=1, colscale='gray', dpi=72, vrange=np.array([0.,1.]), vrangetype='default'):
    nd = array.shape
    fig = plt.figure(figsize=(pixscale*nd[1]/dpi,pixscale*nd[0]/dpi), dpi=dpi)
    ax = fig.add_axes([0,0,1,1], frameon=False, aspect=1)
    ax.axis('off')
    rmin, rmax, arrshow = get_value_range(array, vrange, vrangetype)
    ax.imshow(arrshow, cmap = colscale, origin = 'lower', vmin = rmin, vmax = rmax)
    return fig

#%%
