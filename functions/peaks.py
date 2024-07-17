# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:57:00 2019
@author: ju-bar

Peak functions and functions calculating their jacobians

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
def gauss_2d_jac(x_tuple, x0, y0, a, bxx, bxy, byy, c):
    '''
    Derivatives of
        a * Exp[ - (bxx*bxx + bxy*bxy) * (x-x0)**2 
             - 2 * bxy * (bxx + byy) * (x-x0) * (y-y0)
             - (byy*byy + bxy*bxy) * (y-y0)**2 ] + c
    with respect to parameters x0, y0, a, bxx, bxy, byy, c
    Returns
        numpy.array([dx0, dy0, da, dbxx, dbxy, dbyy, dc]).T
    computed for all M input x_tuple sets, where x0 etc. can
    be arrays of length M
    '''
    (x, y) = x_tuple
    dx = x - x0
    dy = y - y0
    fxx = (bxx * bxx + bxy * bxy)
    fxy = 2 * bxy * (bxx + byy)
    fyy = (byy * byy + bxy * bxy)
    dx2 = dx**2
    dy2 = dy**2
    dxy = dx * dy
    varg = fxx * dx2 + 2 * fxy * dxy + fyy * dy2
    vexp = np.exp(-varg)
    dx0 = a * vexp * 2 * (fxx * dx + fxy * dy)
    dy0 = a * vexp * 2 * (fxy * dx + fyy * dy)
    da = vexp
    dbxx = -2 * a * vexp * (bxx * dx2 + bxy * dxy)
    dbxy = -2 * a * vexp * (bxy * (dx2 + dy2) + (bxx + byy) * dxy)
    dbyy = -2 * a * vexp * (bxy * dxy + byy * dy2)
    dc = np.full(np.size(x), 1., dtype=x_tuple.dtype)
    return np.array([dx0, dy0, da, dbxx, dbxy, dbyy, dc]).T
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
def gauss_2d_round_jac(x_tuple, x0, y0, a, b, c):
    '''
    Derivatives of 
        a * exp[ -b**2 * ((x-x0)**2 + (y-y0)**2) ] + c
    with respect to parameters x0, y0, a, b, c

    Returns
        numpy.array([dx0, dy0, da, db, dc]).T
    computed for all M input x_tuple sets, where x0 etc. can
    be arrays of length M
    '''
    (x, y) = x_tuple
    dx = x - x0
    dy = y - y0
    varg = b**2 * (dx**2 + dy**2)
    vexp = np.exp(-varg)
    dx0 = a * vexp * 2 * b**2 * dx
    dy0 = a * vexp * 2 * b**2 * dy
    da = vexp
    db = -2 * vexp * b * (dx**2 + dy**2)
    dc = np.full(np.size(x), 1., dtype=float)
    return np.array([dx0, dy0, da, db, dc]).T
# %%
def gauss_2d_round_slope(x_tuple, x0, y0, a, b, c, dx, dy):
    '''
    a * exp[-b**2 * ((x-x0)**2 + (y-y0)**2)] + c
        + dx*(x-x0) + dy*(y-y0)
    '''
    (x, y) = x_tuple
    gr = gauss_2d_round(x_tuple, x0, y0, a, b, c)
    return gr + dx*(x-x0) + dy*(y-y0)
def gauss_2d_round_slope_jac(x_tuple, x0, y0, a, b, c, dx, dy):
    '''
    Derivatives of 
        a * exp[-b**2 * ((x-x0)**2 + (y-y0)**2)] + c
        + dx*(x-x0) + dy*(y-y0)
    with respect to parameters x0, y0, a, b, c, dx, dy

    Returns
        numpy.array([dx0, dy0, da, db, dc, ddx, ddy]).T
    computed for all M input x_tuple sets, where x0 etc. can
    be arrays of length M
    '''
    (x, y) = x_tuple
    mx = x - x0
    my = y - y0
    varg = b**2 * (mx**2 + my**2)
    vexp = np.exp(-varg)
    dx0 = a * vexp * 2 * b**2 * mx
    dy0 = a * vexp * 2 * b**2 * my
    da = vexp
    db = -2 * vexp * b * (mx**2 + my**2)
    dc = np.full(np.size(x), 1., dtype=float)
    ddx = mx
    ddy = my
    return np.array([dx0, dy0, da, db, dc, ddx, ddy]).T
# %%
