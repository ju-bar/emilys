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
def gauss_2d(x, y, x0, y0, a, rbxx, rbxy, rbyy, c):
    '''
    a * Exp[ -rbxx*(x-x0)**2 - 2*rbxy*(x-x0)*(y-y0) - rbyy*(y-y0)**2] + c
    '''
    dx = x - x0
    dy = y - y0
    varg = rbxx * dx**2 + 2 * rbxy * dx * dy + rbyy * dy**2
    return a * np.exp(-varg) + c
# %%
def gauss_2d_round(x, y, x0, y0, a, b, c):
    '''
    a * exp[-0.5 * ((x-x0)**2 + (y-y0)**2) / b**2] + c
    '''
    return a * np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / b**2) + c
# %%
def gauss_2d_round_slope(x, y, x0, y0, a, b, c, dx, dy):
    '''
    a * exp[-0.5 * ((x-x0)**2 + (y-y0)**2) / b**2] + c
        + dx*(x-x0) + dy*(y-y0)
    '''
    gr = gauss_2d_round(x, y, x0, y0, a, b, c)
    return gr + dx*(x-x0) + dy*(y-y0)
# %%
