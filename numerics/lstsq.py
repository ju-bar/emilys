# -*- coding: utf-8 -*-
"""
Created on Sun Jul 07 17:48:00 2019
@author: ju-bar

Linear least square algorithms.

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

import numpy as np

class LeastSquareSolution:
    '''
    class LeastSquareSolution

    Data class for communicating the solution x of a linear
    system of equations
        b = A . x
    with more equations, m = length of b
    than unknowns, n = length of x
    Multiple vectors b will cause the same amount of solutions x

    Members:
        x : numpy.ndarray shape(l,n)
            solution vectors
        cov : numpy.ndarray shape(n,n)
            covariance matrix
        chisq : numpy.ndarray shape(l,)
            sum of squared residuals for each solution in x
        m : int
            number of equations
        n : int
            number of parameters
        l : int
            number of solutions
        solver : string
            solving algorithm
        tol : float
            relative singular value strength accepted
        err : string
            error message
        errcode : int
            error code
        
    '''
    def __init__(self):
        self.x = np.array([])
        self.cov = np.array([])
        self.chisq = np.array([])
        self.m = 0
        self.n = 0
        self.l = 0
        self.solver = 'none'
        self.tol = 1.E-8
        self.err = ''
        self.errcode = 0

def linear_lstsq(A, b, tol=1.E-8):
    '''
    Solves the over-determined system of linear equations
        b = A . x
    for x. For the solution we assume pre-weighted equations.
    If weighting is to be applied, please do this outside of
    this routine.

    Parameters:
        A : array_like (M,N), float
            linear system matrix for M equations and N unknowns
        B : array_like (M,) or (L,M), float
            right side vectors
        tol : float
            tolerance for accepting weak singular values relative
            to the strength of the strongest singular value

    Return:
        LeastSquareSolution object
            solution as x member, multiple for L > 1
            covariance matrix as cov
    '''
    sol = LeastSquareSolution()
    mata = np.array(A)
    shpa = mata.shape
    if (len(shpa) != 2):
        sol.errcode = 1
        sol.err = ('input "A" is expected as 2-dimensional array,'+
                   '{:d} dimension(s) found'.format(len(shpa)))
        print('Error in linear_lstsq: ' + sol.err)
        return sol
    sol.m = shpa[0]
    sol.n = shpa[1]
    vb = np.array(b)
    shpb = vb.shape
    if (len(shpb) < 1 or len(shpb) > 2):
        sol.errcode = 2
        sol.err = ('input "b" is expected as 1- or 2-dimensional array,'+
                   '{:d} dimension(s) found'.format(len(shpb)))
        print('Error in linear_lstsq: ' + sol.err)
        return sol
    if (len(shpb) == 2):
        sol.l = shpb[0]
        if (sol.m != shpb[1]):
            sol.errcode = 3
            sol.err = ('dimension 1 of input "b" ({:d}) should be equal'
                       ' to dimension 0 of input A ({:d})'.format(
                        shpb[1],shpa[0]))
            print('Error in linear_lstsq: ' + sol.err)
            return sol
    else: # len(shpb) == 1
        sol.l = 1
        if (sol.m != shpb[0]):
            sol.errcode = 4
            sol.err = ('dimension 0 of input "b" ({:d}) should be equal'
                       ' to dimension 0 of input A ({:d})'.format(
                        shpb[1],shpa[0]))
            print('Error in linear_lstsq: ' + sol.err)
            return sol
    vb = vb.reshape(sol.l, sol.m) # assure shape of solution vectors
    sol.x = np.zeros((sol.l,sol.n)) # prepare solution vectors
    sol.cov = np.zeros((sol.n,sol.n)) # prepare covariance array
    #
    # - implement solver switch here
    sol.solver = 'svd' # using singular value decomposition from numpy.linalg
    matu, diags, matvh = np.linalg.svd(mata)
    matvt = np.transpose(matvh)
    diagstol = diags
    # edit singular values below tol
    smax = np.max(diags) # largest singular value
    stol = sol.tol * smax # singular value threshold
    for i in range(0, sol.n):
        if (diagstol[i] < stol): # singular value below threshold?
            diagstol[i] = 0. #
    #
    # calculate covariance matrix
    wdiag = np.zeros(sol.n)
    w2diag = np.zeros(sol.n)
    for i in range(0, sol.n): # prepare weigths from singular values
        if (diagstol[i] > 0.):
            wdiag[i] = 1. / diagstol[i]
            w2diag[i] = 1. / (diagstol[i]**2)
    for i in range(0, sol.n):
        for j in range(0, sol.n):
            csum = np.dot(matvt[i] * matvt[j], w2diag)
            # csum = 0.
            # for k in range(0, sol.n):
            #     # csum = csum + matvh[k,i] * matvh[k,j] * w2diag[k]
            #     csum = csum + matvt[i,k] * matvt[j,k] * w2diag[k]
            sol.cov[i,j] = csum
            sol.cov[j,i] = csum
    #
    # calculate solutions and residuals
    btmp = np.zeros(sol.m, dtype=float)
    for k in range(0, sol.l): # ... for all right-side vectors in b
        # solutions by back-substitution
        # for j in range(0, sol.m):
        #     bsum = 0.
        #     if (diagstol[j] > 0.):
        #         # for i in range(0, sol.n):
        #         #     bsum = bsum + matu[j,i] * vb[k,i]
        #         # bsum = bsum * wdiag[j]
        #         bsum = np.dot(matu[j], vb[k]) * wdiag[j]
        #     btmp[j] = bsum
        btmp = np.dot(matu, vb[k]) * wdiag
        # for j in range(0, sol.m):
        #     bsum = 0.
        #     for i in range(0, sol.m):
        #         # bsum = bsum + matvh[i,j] * btmp[i]
        #         bsum = bsum + matvt[j,i] * btmp[i]
        #     sol.x[k,j] = bsum
        sol.x[k] = np.dot(matvt, btmp)
        # residuals for the solution by back-projection
        btmp = np.dot(mata, sol.x[k])
        sol.chisq[k] = np.sum((btmp - vb[k])**2)
    #
    return sol