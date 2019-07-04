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
            x : array_like, float, tuple
                position in object plane
            lcoeff : array_like, float
                sequence of projection coefficient interpreted as tuples {[alk0,alk1]}

        You may set the coefficients before calling the project method
        using projection_func_2d.lcoeff = lcoeff.
        '''
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
            for j in range(0, m): # calculate binomial(l+k,l) x0**l * x1**k for all orders
                l, k = self.__lidx_to_coeff[j]
                fpt[j] = fpx[l,0] * fpx[k,1] * self.__lbinom[j]
            ly[i] = np.dot(fpt, self.__lcoeff) # project coefficient lists on binomial term list
        return ly
