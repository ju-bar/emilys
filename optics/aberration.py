# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 20:43:00 2019
@author: ju-bar

Functions and classes handling 2D image aberrations

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
from numba import jit # include compilation support
import numpy as np
import scipy.special

class aberr_axial_func:
    '''
    
    class aberr_axial_func

    Handles the calculation of axial aberration function values.

    aberration_func objects support arbitrary orders of the aberration
    polynomial. The lowest order is 0 and refers to a shift. Spherical
    aberration C_3,0 is of 3rd order.

    '''
    #
    # Coding comment:
    #   Internally, the order of an aberration term is m for an aberration of order m-1.
    #   The rotational symmetry is n, with n = 0 for round aberrations.
    #
    def __init__(self, max_order):
        # maximum polynomial order of the phase terms (max(m) = max_order + 1)
        self.__max_order = max(1, int(max_order)+1) # at least an image shift should be present
        # number of aberration terms
        self.__num_terms = (int)((4 * self.__max_order - self.__max_order % 2 + self.__max_order * self.__max_order) / 4)
        # number of aberration coefficients (2 x terms)
        self.__num_coeff = 2 * self.__num_terms
        # number of indices for partial term orders, must support 0 and max(m) + max(n) = 2 * max(m)
        self.__num_termorders = 1 + 2 * self.__max_order
        # local copy of the last coefficient list
        self.__lcoeff = np.zeros(self.__num_coeff) # preset to 0
        # aberration coefficient list (internal sequence -> (m,n) )
        self.__lterms_idx = np.zeros((self.__num_terms,2), dtype=int) # preset to 0
        # aberration index list ( (m,n) -> internal sequence)
        self.__lidx_terms = np.zeros((self.__max_order+1,self.__max_order+1), dtype=int) # preset to 0
        # aberration term flags
        self.__luse_term = np.full(self.__num_terms, 1, dtype=int) # preset to 1
        # binomial factors (n over k)
        self.__lbinom = np.zeros((self.__num_termorders,self.__num_termorders), dtype=int) # preset to 0
        # coefficient sign list
        self.__lcsgn = np.array([1.,0.,-1.,0.]) #  static list
        # set binomial factors
        for n in range(0, self.__num_termorders):
            for k in range(0, self.__num_termorders):
                self.__lbinom[n,k] = scipy.special.comb(n,k,exact=True)
        # set index list
        l = 0
        for m in range(1, self.__max_order+1):
            for n in range(0, m + 1):
                if 0 == (m+n)%2: # valid orders have even sums of m and n
                    self.__lterms_idx[l,0] = m
                    self.__lterms_idx[l,1] = n
                    self.__lidx_terms[m,n] = l
                    l = l + 1

    def max_order(self):
        '''
        Maximum order of the aberation function polynomial.
        '''
        return self.__max_order

    def num_terms(self):
        '''
        Number of aberration terms.
        '''
        return self.__num_terms

    def num_coeff(self):
        '''
        Number of aberration coeffcients.
        '''
        return self.__num_coeff

    def idx_of_term(self, m, n):
        '''
        List index of the term with polynomial order (m+1) and rotational symmetry n.
        '''
        im = m+1
        if 0 == (im+n)%2 and im <= self.__max_order and n <= im:
            return self.__lidx_terms[im, n]
        return -1 # not a listed term

    def term_of_idx(self, idx):
        '''
        Term order m and rotational symmetry for list index idx returned as tuple (m,n).
        '''
        if idx >= 0 and idx < self.__num_terms:
            return self.__lterms_idx[idx] - [1,0]
        return [0,0] # not a listed term

    def binom(self, n, k):
        return self.__lbinom[n,k]

    def get_lcoeff(self):
        '''
        Returns the internal coefficient list.
        '''
        return self.__lcoeff

    def get_luse_term(self):
        '''
        Returns the internal list of term usage flags.
        '''
        return self.__luse_term

    def set_lcoeff(self, lcoeff):
        '''
        Set internal coefficient list.

        Parameters:
            lcoeff : array_like, float, shape = (self.num_coeff(),) or shorter
                aberration coefficients

        Return:
            integer, number of coefficients set
        '''
        ncf = min(np.size(lcoeff), self.__num_coeff)
        if ncf > 0:
            self.__lcoeff[0:ncf] = lcoeff[0:ncf]
        return ncf

    def set_luse_term(self, luseterm):
        '''
        Set internal coefficient list.

        Parameters:
            luseterm : array_like, int, shape = (self.num_terms(),) or shorter
                aberration term flags

        Return:
            integer, number of coefficient usage flags set
        '''
        nuse = min(np.size(luseterm), self.__num_terms)
        if nuse > 0:
            self.__luse_term[0:nuse] = luseterm[0:nuse]
        return nuse
    
    def chi(self, x_tuple, lcoeff = np.array([]), luseterm = np.array([])):
        '''
        Calculates the aberration function polynomial chi.
        If non-empty lists are provided by parameters 'lcoeff' and 'luseterm',
        respective internal lists will be set from input.
        Empty lists are default.
        
        You may also use methods set_lcoeff(...) and set_luse_term(...) to
        preset the internal arrays, do this with an initial call of chi, or
        with each call of chi.

        Parameters:
            x_typle : array_like, float, shape = (2,)
                coordinate
            lcoeff : array_like, float, shape = (self.num_coeff(),) or shorter
                aberration coefficients
            luseterm : array_like, int, shape = (self.num_terms(),) or shorter
                aberration term flags

        Return:
            float

        '''
        nol = self.__max_order + 1
        wx = x_tuple[0]
        wy = x_tuple[1]
        self.set_lcoeff(lcoeff)
        self.set_luse_term(luseterm)
        w = np.sqrt(wx*wx + wy*wy)
        pwx = 1.
        pwy = 1.
        pw = 1.
        reschi = 0.
        wfield = np.full((nol, 3), 1., dtype=float)
        for i in range(1, nol):
            pwx *= wx
            pwy *= wy
            pw *= w
            wfield[i,0] = pwx
            wfield[i,1] = pwy
            wfield[i,2] = pw
        for k in range(0, self.__num_terms): # loop over aberration terms
            if 0 == self.__luse_term[k]: continue # skip term
            # adding term (m,n)
            m = self.__lterms_idx[k, 0]
            n = self.__lterms_idx[k, 1]
            term = 0.
            for l in range(0, n + 1): # loop l from 0 to n inclusive
                coef = 0.
                tsgn = self.__lcsgn[l%4] # get sign factor for x component of the aberration
                coef = coef + tsgn * self.__lcoeff[2 * k] # access aberration x-coefficient and apply sign
                tsgn = self.__lcsgn[(l+3)%4] # get sign factor for y component of the aberration
                coef = coef + tsgn * self.__lcoeff[1 + 2 * k] # access aberration y-coefficient and apply sign
                j = n - l # get term order
                #   ++ (sgn*ax + sgn*ay) * binom(n over l) * wx**j      * wy**l
                term = term + coef * self.__lbinom[n, l] * wfield[j,0] * wfield[l,1]
            j = m - n # round pre-factor power
            term = term * wfield[j,2] / m # multiply round pre-factor
            reschi = reschi + term # add term to result
        return reschi

    # def grad_chi !
