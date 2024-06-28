# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 20:43:00 2019
@author: ju-bar

Functions and classes handling 2D image aberrations

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
#from numba import njit # include compilation support
import numpy as np

def limitn(n, theta, phase_tol_pi=0.25):
    """
    Calculates the n-th order aberration tolerance for a given
    scattering angle theta in radians and phase tolerane in fractions of Pi.

    Parameters
    ----------
        n : int
            aberration order
                1: defocus, two-fold astigmatism,
                2: coma, 3-fold astigmatism
                3: spherical aberration, star aberration, 4-fold,
                   ...
        theta : float
            scattering angle in radians
        phase_tol_pi : float, optional, default = 0.25
            phase tolerance

    Returns
    -------
        float
            aberration tolerance in units of the electron wavelength / (2*Pi)
            the limit in nm is thus ab.limitn(1, gmax * wl) * wl / (2 * np.pi)
    
    Remarks
    -------
        This has a just-in-time compilation decorator and may take longer
        on the first call.
        Assumes maximum phase shift for an aberration of order n at theta by
        phi_max(n,theta) = 2*Pi / lambda * c * theta**(n+1) / (n+1)
        -> 2*Pi / lambda * c = phi_max * (n+1) * theta**(-n-1)
           2*Pi / lambda * c is returned for phi_max = phase_tol_pi * Pi
    -
    """
    assert n > 0, 'The minimum aberration order supported is n = 1.'
    m = -1 - n
    return np.pi * phase_tol_pi * (n+1) * theta**m

    

class aberr_axial_func:
    '''
    
    class aberr_axial_func

    Handles the calculation of axial aberration function values.

    aberration_func objects support arbitrary orders of the aberration
    polynomial. The lowest order is 0 and refers to a shift. Spherical
    aberration C_3,0 is of 3rd order.

    General form of the aberration function:

    Re[ sum_l=1,N sum_k=0,min(l,N-l) C_l+k-1,l-k * w^k * conjugate(w)^l / (l+k) ],
    with 2d coordinates w = w_x + w_y*j and coefficients C_m-1,n = C_m-1,n,x + C_m-1,n,y*j
    expressed by complex numbers.

    '''
    #
    # Coding comment:
    #   Internally, the order of an aberration term is m for an aberration of order m-1.
    #   The rotational symmetry is n, with n = 0 for round aberrations.
    #
    def __init__(self, max_order):
        # maximum polynomial order of the phase terms (max(m) = max_order + 1)
        self.__max_order = max(1, int(max_order)+1) # at least an image shift should be present
        no = self.__max_order
        # number of aberration terms
        self.__num_terms = (int)((4 * no - no%2 + no**2) / 4)
        nt = self.__num_terms
        # number of indices for partial term orders, must support 0 and max(m) + max(n) = 2 * max(m)
        self.__lcoeff = np.zeros(nt, dtype=complex) # preset to 0
        # aberration term flags
        self.__luse_term = np.full(nt, 1, dtype=int) # preset to 1
        # aberration coefficient list (internal sequence -> (m,n) )
        self.__lterm_of_idx = np.zeros((nt, 2), dtype=int) # preset to 0
        # aberration index list ( (m,n) -> internal sequence)
        self.__lidx_of_term = np.zeros((no + 1, no + 1), dtype=int) # preset to 0
        # set index list
        l = 0
        for m in range(1, no + 1):
            for n in range(0, m + 1):
                if 0 == (m+n)%2: # valid orders have even sums of m and n
                    self.__lterm_of_idx[l,0] = m
                    self.__lterm_of_idx[l,1] = n
                    self.__lidx_of_term[m,n] = l
                    l = l + 1

    @property
    def max_order(self):
        '''
        Maximum order of the aberation function polynomial.
        '''
        return self.__max_order

    @property
    def num_terms(self):
        '''
        Number of aberration terms.
        '''
        return self.__num_terms

    def idx_of_term(self, m, n):
        '''
        List index of the term with polynomial order (m+1)
        and rotational symmetry n.
        '''
        im = m+1
        if 0 == (im+n)%2 and im <= self.max_order and n <= im:
            return self.__lidx_of_term[im, n]
        return -1 # not a listed term

    def term_of_idx(self, idx):
        '''
        Term order m and rotational symmetry for list index idx
        returned as tuple (m,n).
        '''
        if idx >= 0 and idx < self.num_terms:
            return self.__lterm_of_idx[idx] - [1,0]
        return [0,0] # not a listed term

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
            lcoeff : array_like, complex, shape = (self.num_terms,) or shorter
                aberration coefficients

        Return:
            integer, number of coefficients set
        '''
        lcoeff = np.atleast_1d(lc)
        assert lcoeff.dtype == complex, 'lcoeff has to be of complex numbers!'
        ncf = min(np.size(lcoeff), self.num_terms)
        if ncf > 0:
            self.__lcoeff[0:ncf] = lcoeff[0:ncf]
        #return ncf


    @property
    def luse_term(self):
        '''
        Returns a copy of the internal list of term usage flags.
        '''
        return self.__luse_term.copy()

    @luse_term.setter
    def luse_term(self, luse):
        '''
        Set internal coefficient list.

        Parameters:
            luseterm : array_like, int, shape = (self.num_terms,) or shorter
                aberration term flags

        Return:
            integer, number of coefficient usage flags set
        '''
        luse2 = np.atleast_1d(luse)
        assert luse2.dtype == int, 'luse has to be of int numbers!'
        nuse = min(np.size(luse2), self.num_terms)
        if nuse > 0:
            self.__luse_term[0:nuse] = luse2[0:nuse]
        #return nuse
    
    
    def chi(self, x, lcoeff = None, luse = None):
        '''
        Calculates the aberration function polynomial chi.
        If non-empty lists are provided by parameters 'lcoeff' and 'luse',
        respective internal lists of coefficients and use flags will be set
        from input. Empty lists are default.
        
        You may preset the coefficients by self.lcoeff = [ complex values ]
        and the use flags by self.luse_term = [ integer flags ].

        Parameters:
            x : complex
                coordinate
            lcoeff : array_like, complex, shape = (self.num_coeff(),) or shorter
                aberration coefficients
            luse : array_like, int, shape = (self.num_terms(),) or shorter
                aberration term flags

        Return:
            float

        '''
        reschi = 0.
        nol = self.__max_order + 1
        if lcoeff is None:
            i = 1
        else:
            self.lcoeff = lcoeff
        if luse is None:
            i = 1
        else:
            self.luse_term = luse
        px = 1.
        xfn = np.full(nol, 1., dtype=complex)
        for i in range(1, nol):
            px *= x
            xfn[i] = px
        xfc = xfn.conjugate()
        for k in range(0, self.num_terms): # loop over aberration terms
            if 0 == self.__luse_term[k]:
                continue # skip term
            # adding term (m,n)
            m1, n = self.term_of_idx(k)
            m = m1 + 1
            j = int((m + n) / 2)
            l = int((m - n) / 2)
            term = np.real( self.__lcoeff[k] * xfn[l] * xfc[j] ) / m
            reschi = reschi + term # add term to result
        return reschi

    #@jit
    def chi_list(self, lx, lcoeff = None, luse = None):
        '''
        Calculates the aberration function polynomial chi for a list
        of coordinates lx.
        If non-empty lists are provided by parameters 'lcoeff' and 'luse',
        respective internal lists of coefficients and use flags will be set
        from input. Empty lists are default.
        
        You may preset the coefficients by self.lcoeff = [ complex values ]
        and the use flags by self.luse_term = [ integer flags ].

        Parameters:
            lx : array_like, complex
                list of coordinate
            lcoeff : array_like, complex, shape = (self.num_coeff(),) or shorter
                aberration coefficients
            luse : array_like, int, shape = (self.num_terms(),) or shorter
                aberration term flags

        Return:
            np.array, size = size(lx), float

        '''
        nx = np.size(lx)
        if nx == 0:
            return np.array([], dtype=float)
        lc = np.zeros(nx, dtype=float)
        rchi = 0.
        nol = self.__max_order + 1
        if lcoeff is None:
            i = 1
        else:
            self.lcoeff = lcoeff
        if luse is None:
            i = 1
        else:
            self.luse_term = luse
        xfn = np.full(nol, 1., dtype=complex)
        #
        # loop over coordinates
        for i in range(0, nx):
            x = lx[i]
            px = 1.
            for j in range(1, nol):
                px *= x
                xfn[j] = px
            xfc = xfn.conjugate()
            for k in range(0, self.num_terms): # loop over aberration terms
                if 0 == self.__luse_term[k]:
                    continue # skip term
                # adding term (m,n)
                m1, n = self.term_of_idx(k)
                m = m1 + 1
                j = int((m + n) / 2)
                l = int((m - n) / 2)
                term = np.real( self.__lcoeff[k] * xfn[l] * xfc[j] ) / m
                rchi = rchi + term # add term to result
            lc[i] = rchi
        return lc

    def grad_chi(self, x, lcoeff = None, luse = None):
        '''
        Calculates the gradient of aberration function polynomial chi
        in complex number notation.
        If non-empty lists are provided by parameters 'lcoeff' and 'luse',
        respective internal lists of coefficients and use flags will be set
        from input. Empty lists are default.

        You may preset the coefficients by self.lcoeff = [ complex values ]
        and the use flags by self.luse_term = [ integer flags ].
        
        Parameters:
            x : complex
                coordinate
            lcoeff : array_like, complex, shape = (self.num_coeff(),) or shorter
                aberration coefficients
            luse : array_like, int, shape = (self.num_terms(),) or shorter
                aberration term flags

        Return:
            complex

        '''
        gradchi = 0. + 0.j
        nol = self.__max_order + 2
        if lcoeff is None:
            i = 1
        else:
            self.lcoeff = lcoeff
        if luse is None:
            i = 1
        else:
            self.luse_term = luse
        px = 1.
        # initialize complex powers [0., 1., x, x^2, x^3, ...]
        xfn = np.zeros(nol, dtype=complex)
        xfn[1] = 1.
        for i in range(2, nol):
            px *= x
            xfn[i] = px
        xfc = xfn.conjugate() # conjugates of the power list
        for k in range(0, self.num_terms): # loop over aberration terms
            if 0 == self.__luse_term[k]:
                continue # skip term
            # adding term (m,n)
            m1, n = self.term_of_idx(k)
            m = m1 + 1
            m2 = 2*m
            j = int((m + n) / 2)
            l = int((m - n) / 2)
            # remember: index 0 -> 0, 1 -> x^0, 2 -> x^1, 3 -> x^2 ...
            #   therefore l -> l+1, j -> j+1 compared to the series in method chi(...)
            term = self.__lcoeff[k] * xfn[l+1] * xfc[j] * j \
                   + np.conjugate(self.__lcoeff[k]) * xfc[l] * xfn[j+1] * l
            gradchi = gradchi + term / m2 # add term to result
        return gradchi

    #@jit
    def grad_chi_list(self, lx, lcoeff = None, luse = None):
        '''
        Calculates the gradient of aberration function polynomial chi
        in complex number notation for a list of coordinates lx.
        If non-empty lists are provided by parameters 'lcoeff' and 'luse',
        respective internal lists of coefficients and use flags will be set
        from input. Empty lists are default.

        You may preset the coefficients by self.lcoeff = [ complex values ]
        and the use flags by self.luse_term = [ integer flags ].
        
        Parameters:
            lx : array_like, complex
                coordinates
            lcoeff : array_like, complex, shape = (self.num_coeff(),) or shorter
                aberration coefficients
            luse : array_like, int, shape = (self.num_terms(),) or shorter
                aberration term flags

        Return:
            np.array, size=size(lx), complex

        '''
        nx = np.size(lx)
        if nx == 0:
            return np.array([], dtype=complex)
        lg = np.zeros(nx, dtype=complex)
        gradchi = 0. + 0.j
        nol = self.__max_order + 2
        if lcoeff is None:
            i = 1
        else:
            self.lcoeff = lcoeff
        if luse is None:
            i = 1
        else:
            self.luse_term = luse
        # initialize complex powers [0., 1., x, x^2, x^3, ...]
        xfn = np.zeros(nol, dtype=complex)
        xfn[1] = 1.
        #
        # loop over list
        for i in range(0, nx):
            x = lx[i] # get current coordinate
            # prepare powers of x
            px = 1.
            for j in range(2, nol):
                px *= x
                xfn[j] = px
            xfc = xfn.conjugate() # conjugates of the power list
            gradchi = 0. + 0.j # reset current result
            for k in range(0, self.num_terms): # loop over aberration terms
                if 0 == self.__luse_term[k]:
                    continue # skip term
                # adding term (m,n)
                m1, n = self.term_of_idx(k)
                m = m1 + 1
                m2 = 2*m
                j = int((m + n) / 2)
                l = int((m - n) / 2)
                # remember: index 0 -> 0, 1 -> x^0, 2 -> x^1, 3 -> x^2 ...
                #   therefore l -> l+1, j -> j+1 compared to the series in method chi(...)
                term = self.__lcoeff[k] * xfn[l+1] * xfc[j] * j \
                       + np.conjugate(self.__lcoeff[k]) * xfc[l] * xfn[j+1] * l
                gradchi = gradchi + term / m2 # add term to result
            lg[i] = gradchi
        return lg
