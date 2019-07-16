# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 23:04:00 2019
@author: ju-bar

Distribution functions

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
# %%
def bivnorm(x_tuple, x0, y0, a, b, c):
    '''
    Bivariate normal distribution function
    
    p(x,y;a,b,c)
        = Exp[ -1/2 * ( c**2 * (x-x0)**2 
                + 2 * b * (x-x0) * (y-y0) + a**2 * (y-y0)**2 )
                / (a**2 * c**2 - b**2) ] 
                / (2 * np.pi * np.sqrt(a**2 * c**2 - b**2))

    constraints:
        abs(a) > 0
        abs(c) > 0
        a^2 c^2 - b^2 > 0
    '''
    (x, y) = x_tuple
    dx = x - x0
    dy = y - y0
    a2 = a**2
    b2 = b**2
    c2 = c**2
    d1 = a2 * c2 - b2
    d2 = np.reciprocal(np.sqrt(d1) * (2.0 * np.pi))
    d1 = -0.5 * np.reciprocal(d1)
    argx = c2 * dx**2 + 2 * b * dx * dy + a2 * dy**2
    return np.exp( d1 * argx ) * d2
