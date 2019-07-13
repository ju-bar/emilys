# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:57:00 2019
@author: ju-bar

Peak functions

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
# %%
def gauss_2d(x_tuple, x0, y0, a, bxx, bxy, byy, c):
    '''
    a * Exp[ - (bxx*bxx + bxy*bxy) * (x-x0)**2 
             - 2 * bxy * (bxx + byy) * (x-x0) * (y-y0)
             - (byy*byy + bxy*bxy) * (y-y0)**2 ] + c
    '''
    (x, y) = x_tuple
    dx = x - x0
    dy = y - y0
    fxx = (bxx * bxx + bxy * bxy)
    fxy = 2 * bxy * (bxx + byy)
    fyy = (byy * byy + bxy * bxy)
    varg = fxx * dx**2 + 2 * fxy * dx * dy + fyy * dy**2
    return a * np.exp(-varg) + c
# %%
def gauss_2d_round(x_tuple, x0, y0, a, b, c):
    '''
    a * exp[ -b**2 * ((x-x0)**2 + (y-y0)**2) ] + c
    '''
    (x, y) = x_tuple
    dx = x - x0
    dy = y - y0
    varg = b**2 * (dx**2 + dy**2)
    return a * np.exp(-varg) + c
# %%
def gauss_2d_round_slope(x_tuple, x0, y0, a, b, c, dx, dy):
    '''
    a * exp[-b**2 * ((x-x0)**2 + (y-y0)**2)] + c
        + dx*(x-x0) + dy*(y-y0)
    '''
    (x, y) = x_tuple
    gr = gauss_2d_round(x_tuple, x0, y0, a, b, c)
    return gr + dx*(x-x0) + dy*(y-y0)
# %%
