# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 10:14:00 2019
@author: ju-bar
Functions handling geometric distortions in 2D
"""
# %%
from numba import jit # include compilation support
import numpy as np # include numeric functions
# %%
def valid_coeff(n, m):
    '''
    Returns True for a valid combination of n as order of the aberration
    and m as rotational symmetry, False otherwise.

    Parameters:
        n : integer
            order
        m : integer
            rotational symmetry    
    '''
    if n < 0:
        return False
    n2 = (n+1)%2
    if m < n2 or m > n+1 or m%2 != n2:
        return False
    return True
# %%
def validate_coefflist(lcoeff):
    '''
    Returns a list of aberration coefficients, which is a
    valid version of the input list lcoeff.

    Assumes lcoeff.shape = (N,4)

    '''
    n0 = np.shape(lcoeff)[0]
    nv = 0
    lvalid = lcoeff * 0.
    for l in range(0,n0):
        if valid_coeff(int(lcoeff[l,0]),int(lcoeff[l,1])):
            lvalid[nv] = lcoeff[l] # valid aberration
            nv = nv + 1
    return lvalid[0:nv]
# %%
def dist_func1(x, params):
    '''

    1st order distortion: x -> y0 + mt.x
    on arbitrary dimensions

    Parameters:
        x : numpy.ndarray, size = N
            input coordinate
        params: numpy.ndarray, size = N*(N+1) or less
            coefficients for shift and affine transformation matrix
            with N = np.size(x), params is expected to have N*(N+1)
            elements. The first N elements are uses as shift vector.
            The N*N elements following are used as transformation
            matrix.

    Return:
        numpy.ndarray, size = N

    Remarks:
        Input arrays are flattened before interpretation.
        A flattened array is returned as result.
        Insufficient number of parameter coefficients will
        be interpreted as unity operations.

    '''
    lx = np.ravel(x)
    n = np.size(lx)
    nt = n*(n+1)
    lprm = np.ravel(params)
    nprm = np.size(lprm)
    if nprm < n:
        # insufficient transformation parameters -> no shift and transform
        return lx
    if nprm < nt:
        # only shift operation
        return lx + lprm[0:n]
    # return result after transform + shift
    return np.dot(lprm[n:nt].reshape(n,n), lx) + lprm[0:n]
# %%
def dist2d_func1(x, shift, transform):
    '''

    1st order distortion: x -> y0 + mt.x
    on 2 dimensions

    Parameters:
        x : numpy.ndarray, size = 2
            input coordinate
        shift: numpy.ndarray, size = 2
            shift vector
        transform: numpy.ndarray, shape = (2,2)
            transformation matrix

    Return:
        numpy.ndarray, size = 2

    Remarks:
        This is only slightly faster than dist_func1.

    '''
    return np.dot(transform, x) + shift
# %%
def dist2d_term(n, m, x_tuple):
    '''

    Calculates the 2d basis terms for the aberration of order n and
    rotational symmetry m at position x_tuple in the object plane.

    Parameters:
        n : integer
            order
        m : integer
            rotational symmetry
        x_tuple : numpy.ndarray ([x0,x1])
            object plane position

    Returns:
        numpy.ndarray ([b0,b1])
            basis terms

    '''
    t = np.array([0.,0.])
    # go on here
    return t
# %%
def dist2d_func(x, params, checkinput=True):
    '''

    M order distortion: x -> f(x)
    on 2 dimensions

    Parameters:
        x : numpy.ndarray, size = 2
            input coordinate
        params : numpy.ndarray, size = 4*l
            distortion coefficients as list of 4 tuples
            [[ni,mi,ai0,ai1]]
            ni : aberration order
            mi : aberration rotational symmetry in range((ni+1)%2,2+ni+(ni+1)%2,2)
            ai0 : x-coefficient
            ai1 : y-coefficient
        checkinput : bool, default = True
            check validity of input parameters

    Return:
        numpy.ndarray, size = 2

        May return with numpy.array([0.,0.]) in case of an error.

    Remarks:
        The return vector is calculated as with
        d = a01 + a10 * x + a12 * x' + a21 * x**2 + a23 * x'**2 
                + a30 * x**2 * x' + a32 * x * x'**2 + a34 * x'**3 + ...
        when taking the coefficients anm and vectors x, d as numbers in
        the complex plane, with x' the complex conjugate of x.

        'params' should include [1,0,M,0] as magnification M, 
        with M = 1 unity operation.

        Repeated definition of the same aberration act cumulatively.
        Sum up the coefficients first to avoid extra computation time.

    '''
    lprm = np.ravel(params)
    nprm = np.size(lprm) # number of coefficients
    na0 = int((nprm - nprm%4) / 4) # number of aberration coefficients
    if na0 > 0: # distortions defined
        lacf0 = lprm[0:4*na0].reshape(na0,4) # reshape input to 4-tuples
    else:
        lacf0 = np.array([])
    if checkinput:
        if np.size(x) != 2: # invalid dimension of the input coordinate
            return np.array([0.,0.])
        # check aberration parameter list
        lacf = validate_coefflist(lacf0)
        na = np.shape(lacf)[0]
    else: # use input as is
        lacf = lacf0
        na = na0
    vx = x[0:2]
    vy = np.array([0.,0.])
    if na > 0:
        for l in range(0, na): # loop over listed aberrations
            vy = vy + np.dot(lacf[l,2:4], dist2d_term( int(lacf[l,0]), int(lacf[l,1]), vx)) # sum up the terms
    return vy