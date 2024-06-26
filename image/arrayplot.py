# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:42:00 2019
@author: ju-bar

Array data plotting wrapper

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
# %%
import matplotlib.pyplot as plt # include mathematic plotting
import numpy as np # include numeric functions
# %%
def get_value_range(array, vrange=np.array([0.,1.]), vrangetype='default'):
    """
    Calculates the range of values of an array according to range type
    and range limit parameters. May return array data with modified
    values.

    Parameters
    ----------
    array : array of 2 dimensions
        array data intended for plotting
    vrange : array shape(2,)
        range limits
    vrangetype : string
        value range setting mode:
            'default' : min. to max. of array values,
                        modified relatively by vrange
            'direct'  : plots from vrange[0] to vrange[1]
            'modulo'  : plots from vrange[0] to vrange[1],
                        modifies array -> arrshow with by clipping
                        all values into this range using a modulo
                        operation
            'zerosym' : symmetric around zero covering max. absolute
                        array value, modified relatively by vrange

    Return
    ------
    array
        [rmin, rmax, arrshow]
        rmin : float
            minimum value to be plotted
        rmax : float
            maximum value to be plotted
        arrshow : array
            modified values to be plotted in range [rmin, rmax]
    """
    vmin = np.amin(array)
    vmax = np.amax(array)
    rmin = vmin + vrange[0] * (vmax - vmin)
    rmax = vmin + vrange[1] * (vmax - vmin)
    arrshow = array
    if vrangetype == 'zerosym': # set value range symmetric around zero
        if vmax < 0:
            vmax = np.abs(vmin)
        else:
            vmin = -vmax
        rmin = vmin + vrange[0] * (vmax - vmin)
        rmax = vmin + vrange[1] * (vmax - vmin)
    elif vrangetype == 'direct': # set the value range directly from vrange
        rmin = vrange[0]
        rmax = vrange[1]
    elif vrangetype == 'modulo': # use modulo on the value range
        rmin = vrange[0]
        rmax = vrange[1]
        arrshow = np.mod(array - vrange[0], vrange[1] -vrange[0]) + vrange[0]
    return [rmin, rmax, arrshow]
# %%
def arrayplot2d(array, pixscale=1, colscale='gray', dpi=72, 
                vrange=np.array([0.,1.]), vrangetype='default', hide=False):
    """
    Uses matplotlib.pyplot to plot a pixel-true image of the input 'array'.

    Parameters
    ----------
        array : array shape(n,m)
            input data to plot as image
        pixscale : int or float : default 1
            number of image dots per array element
        colscale : string : default 'gray'
            color palette identifier
        dpi : int : default 72
            dots per inch for image output
        vrange : array shape(2,) : default [0.,1.]
            value range limits
        vrangetype : string : default 'default'
            value range setting mode:
                'default' : min. to max. of array values,
                            modified relatively by vrange
                'direct'  : plots from vrange[0] to vrange[1]
                'modulo'  : plots from vrange[0] to vrange[1],
                            modifies array -> arrshow with by clipping
                            all values into this range using a modulo
                            operation
                'zerosym' : symmetric around zero covering max. absolute
                            array value, modified relatively by vrange
        hide : boolean
            flag to not plot the image on screen, default: False

    Return
    ------
        [matplotlib.pyplot.figure, matplotlib.pyplot.axes]
            figure and axes object used for plotting the array

    """
    nd = array.shape
    fig = plt.figure(figsize=(pixscale*nd[1]/dpi,pixscale*nd[0]/dpi), dpi=dpi)
    ax = fig.add_axes([0,0,1,1], frameon=False, aspect=1)
    ax.axis('off')
    rmin, rmax, arrshow = get_value_range(array, vrange, vrangetype)
    ax.imshow(arrshow, cmap = colscale, origin = 'lower', vmin = rmin,
              vmax = rmax)
    if hide:
        plt.close()
    return [fig,ax]

#%%
