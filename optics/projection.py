# -*- coding: utf-8 -*-
"""
Created on Wed Jul 04 21:43:00 2019
@author: ju-bar

Functions and classes handling the projection from a 2D object plane
into a 2D image plane including distortions.

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
from numba import jit # include compilation support
import numpy as np
from scipy.special import comb

class projection_func_2d:
    '''
    
    class projection_func_2d

    Handles the calculation of a projection transforming a tuple
    (x0,x1) into a projected tuple (y0,y1).

    projection_func objects support arbitrary polynomial orders of
    distortions. The lowest order (0) is a rigid shift, the first
    order corresponds to a 2x2 matrix projection. Initializing
    with more orders than needed will waste computation power.

    The projection is parameterized by a list of coefficients
    [{alk0},{alk1}] defining the projection according to the
    polynomial series

    [y0,y1] = [a000,a001] + [[a100,a010],[a101,a011]].[x0,x1]
        + [[a200,a110,a020],[a201,a111,a021]].[x0**2,2*x0*x1,x1**2]
        + [[a300,a210,a120,a030],[a301,a211,a121,a031]].[x0**3,3*x0**2*x1,3*x0*x1**2,x1**3]
        + ...
        = sum_l sum_k binomial(l+k,l) * x0**l * x1**k * [alk0 , alk1]

    The linear coefficients are listed in lcoeff in the sequence
    [...,[a100,a101],[a010,a011],...]. A linear distortion matrix
    corresponding to [[a100,a010],[a101,a011]] must therefore be
    entered in transposed form.

    '''
    #
    def __init__(self, max_order):
        # maximum polynomial order of the projection function per dimension
        self.__max_order = max(1, int(max_order)) # at least the 1st order is present
        # number of polynomial terms = number of coefficients per dimension
        self.__num_terms = int((self.max_order + 1) * (self.max_order + 2) / 2 )
        # internal coefficient list
        self.__lcoeff = np.zeros((self.num_terms, 2), dtype=float) # preset to 0
        # coefficient to index list (l,k) -> idx, -1 = no coefficient (not all are used), READ BUT DO NOT WRITE!
        self.__lcoeff_to_idx = np.full((self.max_order + 1,self.max_order + 1), -1, dtype=int)
        # index to coefficient list idx -> (l,k), READ BUT DO NOT WRITE!
        self.__lidx_to_coeff = np.zeros((self.num_terms, 2), dtype=int)
        # binomial factor list idx -> binomial(l+k,l), READ BUT DO NOT WRITE!
        self.__lbinom = np.zeros(self.num_terms, dtype=int)
        # fill the static lists
        idx = 0
        for n in range(0, self.max_order + 1):
            for l in range(n,-1,-1):
                self.__lcoeff_to_idx[n-l,l] = idx
                self.__lidx_to_coeff[idx,0] = l
                self.__lidx_to_coeff[idx,1] = n - l
                self.__lbinom[idx] = comb(n,l,exact=True)
                idx += 1

    def reinitialize(self, max_order=1):
        '''
        Re-initializes to object to the given projection order (min 1 = linear)

        Parameters:
            max_order : int
                order of the projection polynomial
        '''
        self.__init__(max_order)

    @property
    def max_order(self):
        '''
        Maximum order of the polynomial.
        '''
        return self.__max_order

    @property
    def num_terms(self):
        '''
        Number of terms.
        '''
        return self.__num_terms

    @property
    def lcoeff(self):
        '''
        Returns a copy of the internal coefficient list.
        '''
        return self.__lcoeff.copy()

    @lcoeff.setter
    def lcoeff(self, lc):
        '''
        Set internal coefficient list.

        Parameters:
            lcoeff : array_like, complex, shape = (self.num_terms,2) or shorter
                polynomial coefficients

        Return:
            integer, number of coefficients set
        '''
        lc1 = np.atleast_1d(lc)
        assert lc1.dtype in [float,int], 'lcoeff needs numbers!'
        nc1 = np.size(lc1)
        if nc1 > 1:
            nc2 = int((nc1 - nc1%2) / 2)
            lc2 = lc1.flatten()
            self.__lcoeff[0:nc2] = lc2[0:2*nc2].reshape(nc2, 2)

    def set_coeff_idx(self, idx, alk0 = None, alk1 = None):
        '''
        Sets coefficients of the 2 dimensions at index idx
        of the current coefficient list

        Parameters:
            idx : int
                list index
            alk0, alk1 : float
                polynomial term coefficients
        '''
        if idx >= 0 and idx < self.num_terms:
            if alk0 != None and alk0 in [float, int]:
                self.__lcoeff[idx,0] = alk0
            if alk1 != None and alk1 in [float, int]:
                self.__lcoeff[idx,0] = alk0

    def set_coeff_lk(self, l, k, alk0 = None, alk1 = None):
        '''
        Sets coefficients of the 2 dimensions for the term x0**l x1**k 

        Parameters:
            l, k : int
                polynomial term exponents
            alk0, alk1 : float
                polynomial term coefficients
        '''
        if l >= 0 and k >= 0 and l+k <= self.max_order:
            idx = self.__lcoeff_to_idx[k,l]
            self.set_coeff_idx(idx, alk0, alk1)

    def project(self, x, lcoeff = None):
        '''
        Projects x with a 2d polynomial.

        Parameters:
            x : array_like, float, tuple, shape = (N,2)
                position in object plane [...,[xi0,xi1],...]
            lcoeff : array_like, float, optional
                sequence of projection coefficient interpreted as tuples {[alk0,alk1]}

        Return:
            numpy.ndarray
                [...,[yi0,yi1],...]
                list of projected positions as array of shape(N, 2) with
                N = number of tuples in parameter x = numpy.size(x)/2

        Remarks:
            You may set the coefficients before calling the project method
            using projection_func_2d.lcoeff = lcoeff.

            The function calculates for each tuple [xi0,xi1]:
            [yi0,yi1] = sum_j=1,M [alk0, alk1] * tlk(xi0,xi1),
            with tlk(xi0,xi1) = binomial(l+k,l) * x0**l * x1**k
            and an internal listing j :-> l,k for each j in range(0, M).
            M = projection_func_2d.num_terms

        '''
        if lcoeff != None: self.lcoeff = lcoeff # set new coefficients if present
        nx1 = np.size(x)
        assert 0 == nx1%2 and nx1 > 0, 'project expects an even number of coordinates'
        nx = int(nx1 / 2)
        lx = np.reshape(x,(nx,2))
        ly = np.zeros((nx,2), dtype=float) # prepare result arrays
        n = self.max_order
        m = self.num_terms
        fpx = np.full((n+1, 2), 1., dtype=float)
        fpt = np.zeros(m, dtype=float)
        for i in range(0, nx): # for all positions
            pwx = lx[i]
            for l in range(1, n+1): # calculate x0**l and x1**l for l = 1 .. n
                fpx[l] = pwx
                pwx = pwx * lx[i]
            for j in range(0, m): # calculate binomial(l+k,l) x0**l * x1**k for all terms
                l, k = self.__lidx_to_coeff[j]
                fpt[j] = fpx[l,0] * fpx[k,1] * self.__lbinom[j]
            ly[i] = np.dot(fpt, self.__lcoeff) # project coefficient lists on binomial term list
        return ly

    def project_deriv_lcoeff(self, x):
        '''
        2d projection function derivatives with respect to all projection
        parameters (lcoeff). Since the projection is a linear function of
        the parameters, the result depends on x only.

        Parameters:
            x : array_like, float, tuple, shape = (N,2)
                position in object plane [...,[xi0,xi1],...]

        Return:
            numpy.ndarray
                [...,[...,[[dyi0/dalk0, dyi1/dalk0],[dyi0/dalk1, dyi1/dalk1]],...],...]
                list of derivatives as numpy.ndarray of shape(N, M, 2, 2) with
                N = number of tuples in parameter x
                M = number of projection terms tlk(xi0,xi1) = binomial(l+k,l) * x0**l * x1**k

        Remarks:
            The output of this method may be used as Jacobian for least-squares methods.
            Since
                dyi0/dalk0 = dyi1/dalk1 = tlk(xi0,xi1), and
                dyi1/dalk0 = dyi0/dalk1 = 0,
            the 2 x 2 items of the output are [[1,0],[0,1]] * tlk(xi0,xi1).
            This redundancy produces the correct shape of the derivative array for the shape
            (M,2) of the parameter list lcoeff. It consumes 4 times more memory than really needed,
            but this should be acceptable since M is usually not very large (<100).
        '''
        nx1 = np.size(x)
        assert 0 == nx1%2 and nx1 > 0, 'project expects an even number of coordinates'
        nx = int(nx1 / 2)
        lx = np.reshape(x,(nx,2))
        n = self.max_order
        m = self.num_terms
        ldy = np.zeros((nx,m,2,2), dtype=float) # prepare result arrays
        fpx = np.full((n+1, 2), 1., dtype=float)
        fpt = np.zeros(m, dtype=float)
        mid = np.array([[[1.],[0.]],[[0.],[1.]]])
        for i in range(0, nx): # for all positions
            pwx = lx[i]
            for l in range(1, n+1): # calculate x0**l and x1**l for l = 1 .. n
                fpx[l] = pwx
                pwx = pwx * lx[i]
            for j in range(0, m): # calculate tlk(x) = binomial(l+k,l) x0**l * x1**k for all terms
                l, k = self.__lidx_to_coeff[j]
                fpt[j] = fpx[l,0] * fpx[k,1] * self.__lbinom[j]
            ldy[i] = np.dot(mid,[fpt]).T # writes [...,[[tlk(x),0],[0,tlk(x)]],...]
        return ldy

    def project_deriv_x(self, x, lcoeff=None):
        '''
        2d projection function derivatives with respect to positions x,
        evaluated for all x.

        Parameters:
            x : array_like, float, tuple, shape = (N,2)
                position in object plane [...,[xi0,xi1],...]
            lcoeff : array_like, float, optional
                sequence of projection coefficient interpreted as tuples {[alk0,alk1]}

        Return:
            numpy.ndarray
                [...,[[dyi0/dx0,dyi1/dx0],[dyi0/dx1,dyi1/dx1]],...]
                list of derivatives as array of shape(N, 2, 2) with
                N = number of tuples in parameter x
        
        Remarks:
            You may set the coefficients before calling the project_deriv_x method
            using projection_func_2d.lcoeff = lcoeff.

            The output of this method may be used as Jacobian for least-squares methods.
        
        '''
        if lcoeff != None: self.lcoeff = lcoeff # set new coefficients if present
        nx1 = np.size(x)
        assert 0 == nx1%2 and nx1 > 0, 'project expects an even number of coordinates'
        nx = int(nx1 / 2)
        lx = np.reshape(x,(nx,2))
        n = self.max_order
        m = self.num_terms
        ldy = np.zeros((nx,2,2), dtype=float) # prepare result arrays
        fpx = np.full((n+1, 2), 1., dtype=float)
        fpt = np.zeros((m,2), dtype=float) # prepare derivative terms
        for i in range(0, nx): # for all positions
            pwx = lx[i]
            for l in range(1, n+1): # calculate x0**l and x1**l for l = 1 .. n
                fpx[l] = pwx
                pwx = pwx * lx[i]
            for j in range(0, m): # calculate binomial(l+k,l) [l * x0**(l-1) * x1**k, k * x0**l * x1**(k-1)] for all terms
                l, k = self.__lidx_to_coeff[j]
                fpt0 = 0.
                if l>0: # catch derivative of constant (no element of fpx[l-1] for l-1 = -1)
                    fpt0 = l * fpx[l-1,0] * fpx[k,1] * self.__lbinom[j] # (dtj/dx0)
                fpt1 = 0.
                if k>0: # catch derivative of constant (no element of fpx[k-1] for k-1 = -1)
                    fpt1 = k * fpx[l,0] * fpx[k-1,1] * self.__lbinom[j] # (dtj/dx1)
                fpt[j] = np.array([fpt0,fpt1]) # set derivative tuple of term j
            # project coefficient lists on binomial term derivative list
            ldy[i] = np.dot(fpt.T, self.__lcoeff) # (2,m).(m,2) -> (2,2): [[dyi0/dx0,dyi1/dx0],[dyi0/dx1,dyi1/dx1]]
        return ldy
