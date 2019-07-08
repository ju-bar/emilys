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
from scipy.optimize import minimize
from emilys.numerics.lstsq import linear_lstsq as llstsq

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
        # coefficient / term use flags (functions will return zeros for not flagged terms)
        self.__luse_term = np.full(self.num_terms, 1, dtype=int) # preset to 1
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

    def get_lk(self, idx):
        '''
        Returns the polynomial powers (l,k) for the term of given index.
        '''
        if idx >=0 and idx < self.num_terms:
            return self.__lidx_to_coeff[idx]
        return np.array([])

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
            lcoeff : array_like, float, shape = (self.num_terms,2) or shorter
                polynomial coefficients

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
            if alk0 != None:
                self.__lcoeff[idx,0] = alk0
            if alk1 != None:
                self.__lcoeff[idx,1] = alk1

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

    @property
    def luse(self):
        '''
        Returns a copy of the internal usage flag list.
        '''
        return self.__luse_term.copy()

    @luse.setter
    def luse(self, lu):
        '''
        Set internal usage flag list.

        Parameters:
            lu : array_like, int, shape = (self.num_terms,) or shorter
                polynomial term flags

        '''
        lu1 = np.atleast_1d(lu)
        assert lu1.dtype in [float,int], 'luse needs numbers!'
        nu = np.size(lu1)
        if nu > 0:
            lu2 = lu1.flatten().astype(int)
            self.__luse_term[0:nu] = lu2[0:nu]

    def set_use_idx(self, idx, use=1):
        '''
        Sets useage flag for index idx of the current coefficient list (1 by default)

        Parameters:
            idx : int
                list index
            use : int, default: 1
                polynomial term flag
        '''
        if idx >= 0 and idx < self.num_terms:
            self.__luse_term[idx] = int(use)

    def set_use_lk(self, l, k, use=1):
        '''
        Sets useage flag for the term x0**l x1**k (1 by default)

        Parameters:
            l, k : int
                polynomial term exponents
            use : int
                polynomial term flag
        '''
        if l >= 0 and k >= 0 and l+k <= self.max_order:
            idx = self.__lcoeff_to_idx[k,l]
            self.set_use_idx(idx, use)

    def project(self, x, lcoeff = np.array([]), luse=np.array([])):
        '''
        Projects x with a 2d polynomial.

        Parameters:
            x : array_like, float, tuple, shape = (N,2)
                position in object plane [...,[xi0,xi1],...]
            lcoeff : array_like, float, optional
                sequence of projection coefficient interpreted as tuples {[alk0,alk1]}
            luse : array_like, int, optional
                polynomial term flags

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
        if np.size(lcoeff) > 0: self.lcoeff = lcoeff # set new coefficients if present
        if np.size(luse) > 0: self.luse = luse # set flags
        nx1 = np.size(x)
        assert 0 == nx1%2 and nx1 > 0, 'project expects an even number of coordinates'
        nx = int(nx1 / 2)
        lx = np.reshape(x,(nx,2))
        ly = np.zeros((nx,2), dtype=float) # prepare result arrays
        n = self.max_order
        m = self.num_terms
        fpx = np.full((n+1, 2), 1., dtype=float)
        fpt0 = np.zeros(m, dtype=float)
        for i in range(0, nx): # for all positions
            pwx = lx[i]
            for l in range(1, n+1): # calculate x0**l and x1**l for l = 1 .. n
                fpx[l] = pwx
                pwx = pwx * lx[i]
            fpt = fpt0 # reset terms
            for j in range(0, m): # calculate binomial(l+k,l) x0**l * x1**k for all terms
                if 0 == self.__luse_term[j]: continue # skip term j
                l, k = self.__lidx_to_coeff[j]
                fpt[j] = fpx[l,0] * fpx[k,1] * self.__lbinom[j]
            ly[i] = np.dot(fpt, self.__lcoeff) # project coefficient lists on binomial term list
        return ly

    def project_deriv_lcoeff(self, x, luse=np.array([])):
        '''
        2d projection function derivatives with respect to all projection
        parameters (lcoeff). Since the projection is a linear function of
        the parameters, the result depends on x only.

        Parameters:
            x : array_like, float, tuple, shape = (N,2)
                position in object plane [...,[xi0,xi1],...]
            luse : array_like, int, optional
                polynomial term flags

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
        if np.size(luse) > 0: self.luse = luse # set flags
        nx1 = np.size(x)
        assert 0 == nx1%2 and nx1 > 0, 'project expects an even number of coordinates'
        nx = int(nx1 / 2)
        lx = np.reshape(x,(nx,2))
        n = self.max_order
        m = self.num_terms
        ldy = np.zeros((nx,m,2,2), dtype=float) # prepare result arrays
        fpt0 = np.zeros(m, dtype=float)
        fpx = np.full((n+1, 2), 1., dtype=float)
        mid = np.array([[[1.],[0.]],[[0.],[1.]]])
        for i in range(0, nx): # for all positions
            pwx = lx[i]
            for l in range(1, n+1): # calculate x0**l and x1**l for l = 1 .. n
                fpx[l] = pwx
                pwx = pwx * lx[i]
            fpt = fpt0 # reset terms
            for j in range(0, m): # calculate tlk(x) = binomial(l+k,l) x0**l * x1**k for all terms
                if 0 == self.__luse_term[j]: continue # skip term j
                l, k = self.__lidx_to_coeff[j]
                fpt[j] = fpx[l,0] * fpx[k,1] * self.__lbinom[j]
            ldy[i] = np.dot(mid,[fpt]).T # writes [...,[[tlk(x),0],[0,tlk(x)]],...]
        return ldy

    def project_deriv_x(self, x, lcoeff=np.array([]), luse=np.array([])):
        '''
        2d projection function derivatives with respect to positions x,
        evaluated for all x.

        Parameters:
            x : array_like, float, tuple, shape = (N,2)
                position in object plane [...,[xi0,xi1],...]
            lcoeff : array_like, float, optional
                sequence of projection coefficient interpreted as tuples {[alk0,alk1]}
            luse : array_like, int, optional
                polynomial term flags

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
        if np.size(lcoeff) > 0: self.lcoeff = lcoeff # set new coefficients if present
        if np.size(luse) > 0: self.luse = luse
        nx1 = np.size(x)
        assert 0 == nx1%2 and nx1 > 0, 'project expects an even number of coordinates'
        nx = int(nx1 / 2)
        lx = np.reshape(x,(nx,2))
        n = self.max_order
        m = self.num_terms
        ldy = np.zeros((nx,2,2), dtype=float) # prepare result arrays
        fpx = np.full((n+1, 2), 1., dtype=float)
        fptz = np.zeros((m,2), dtype=float) # prepare derivative terms
        for i in range(0, nx): # for all positions
            pwx = lx[i]
            for l in range(1, n+1): # calculate x0**l and x1**l for l = 1 .. n
                fpx[l] = pwx
                pwx = pwx * lx[i]
            fpt = fptz # reset terms
            for j in range(0, m): # calculate binomial(l+k,l) [l * x0**(l-1) * x1**k, k * x0**l * x1**(k-1)] for all terms
                if 0 == self.__luse_term[j]: continue # skip term j
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

    # --------------------- #
    #                       #
    # least squares methods #
    #                       #
    # --------------------- #

    def __getweights(self, ysigm):
        '''
        Returns an array of weights for an array of error estimates ysigm
        using a mean : individual mixture

        This method is private and should only be called from fit_* methods.

        Parameter:
            ysigm : numpy.array (m,)
                error estimates

        Return:
            weights : numpy.array (m,)
                weights
        '''
        m = np.size(ysigm)
        if (m == 0): return np.array([])
        ysm = np.mean(ysigm)
        if (ysm <= 0.): return np.full(m, 1., dtype=float)
        yss = np.std(ysigm)
        fm = np.min([0.5, yss / ysm]) # mean fraction in weight model
        ys = ysigm * (1. - fm) + fm * ysm
        return np.reciprocal(ys)


    # [x0, lcoeff[1:]] -> prm
    def __getprm_fitx0(self, x0, lcoeff):
        '''
        Return a parameter array for the fitx0 method.
        
        This method is private and should only be called from fit_x0_lcoeff.

        Parameters:
            x0 : numpy.ndarray, shape (2,)
            lcoeff: numpy.ndarray, shape (self.num_terms,2)
        
        Return:
            numpy.ndarray, shape (n,)
        '''
        # all inputs are expected as numpy.ndarray
        luse = self.__luse_term.copy()
        nprm = int(2 * (1 + np.sum(luse[1:])))
        lprm = np.zeros(nprm, dtype=float)
        if np.size(x0) > 1:
            lprm[0:2] = x0[0:2]
        l = 1
        for i in range(1,self.num_terms):
            if luse[i] == 1:
                lprm[2*l:2*l+2] = lcoeff[i,0:2]
                l = l + 1
        return lprm

    # prm -> [x0, lcoeff[1:]]
    def __getcoeff_fitx0(self, lprm):
        '''
        Returs x0 and an lcoeff array for the fitx0 method.

        This method is private and should only be called from fit_x0_lcoeff.
        
        Parameters:
            lprm: numpy.ndarray shape(n,)
            luse: numpy.ndarray shape(self.num_terms,)
        
        Return:
            [ numpy.ndarray shape(2,) , numpy.ndarray shape(self.num_terms,2)
        '''
        luse = self.__luse_term.copy()
        x0 = lprm[0:2]
        lcoeff = np.zeros((self.num_terms, 2))
        l = 1
        for i in range(1,self.num_terms):
            if luse[i] == 1:
                lcoeff[i] = lprm[2*l:2*l+2]
                l = l + 1
        return [x0, lcoeff]

    def __getderiv_fitx0(self, ldx0, ldcoeff):
        '''
        Returns a parameter derivative array for the fitx0 method.

        This method is private and should only be called from fit_x0_lcoeff.
        
        Parameters:
            ldx0 : numpy.ndarray, shape (m,2,2)
            ldcoeff: numpy.ndarray, shape (m,self.num_terms,2,2)
        
        Return:
            numpy.ndarray, shape (m,n/2,2,2), n = total number of parameters
        '''
        luse = self.__luse_term.copy()
        nh = int(1 + np.sum(luse[1:]))
        m = ldx0.shape[0]
        ldprm = np.zeros((m,nh,2,2), dtype=float)
        ldprm[:,0:1,:,:] = ldx0.reshape(m,1,2,2)
        l = 1
        for i in range(1, self.num_terms):
            if luse[i] == 1:
                ldprm[:,l,:,:] = ldcoeff[:,i,:,:]
                l = l + 1
        return ldprm


    # define a function calculating squared differences between measured and
    # modelled peak positions using the 2d projection function
    # prm = [x0, dfobj.lcoeff[1:]]
    def __sdevproj_x0(self,x): # assumes shape(x) = (n,)
        x0, lc = self.__getcoeff_fitx0(x)
        #lc[2:] = x[2:] # copy input
        self.lcoeff = lc # set input
        lx = self.__xdata_fitx0 + x0 # x[0:2] # (m,2) : dxi = xi + x0
        lwf = self.__wdata_fitx0.flatten() # (2*m) : wi
        dpmf = (self.project(lx) - self.__ydata_fitx0).flatten() # (2*m,) : dyi = p(dxi) - yi : [dyi0,dyi1,dy10,dy11,dy20,dy21,...,dyn0,dyn1]
        return np.dot(dpmf, dpmf * lwf) #/ np.sum(lwf) # s = sum_i=1,m[ sum_a=0,1[ dyi**2 * wi ] ]
    # ... and define a Jacobian
    def __dsdevproj_x0(self,x): # assumes shape(x) = (n,)
        x0, lc = self.__getcoeff_fitx0(x)
        #lc = np.zeros(self.num_terms*2)
        #lc[2:] = x[2:] # copy input
        self.lcoeff = lc # set input
        lx = self.__xdata_fitx0 + x0 # x[0:2] # (m,2) : dxi = xi + x0
        lwf = self.__wdata_fitx0.flatten() # (2*m) : wi
        dpm = (self.project(lx) - self.__ydata_fitx0) # (m,2) : dyi = p(dxi) - pi : [[dy00,dy01],[dy10,dy11],[dy20,dy21],...,[dym0,dym1]
        dx0 = self.project_deriv_x(lx) # (m,2,2) : D[dyi,x0] [...,[[ddyi0/dx00, ddyi1/dx00], [ddyi0/dx01 + ddyi1/dx01],...]
        #sdx0 = dx0.shape
        #dx0s = np.reshape(dx0,(sdx0[0],1,sdx0[1],sdx0[2])) # (m,1,2,2)
        dcoeff = self.project_deriv_lcoeff(lx) # (m,n/2,2,2) : D[dyi,aj] = [...,[[ddyi0/daj0,ddyi1/daj0],[ddyi0/daj1,ddyi1/daj1]],...]
        #dp = dcoeff # (m,n/2,2,2)
        #sdp = dp.shape
        #dp[:,0,:,:] = dx0s[:,0,:,:] # (m,n/2,2,2) (replace y shift derivatives by x shift dervatives)
        # ... combine dp and dpm to # (n/2,2) : [...,sum_i[dyi0*ddyi0/daj0 + dyi1*ddyi1/daj0, dyi0*ddyi0/daj1 + dyi1*ddyi1/daj1],...]
        dp = self.__getderiv_fitx0(dx0, dcoeff) # (m,n/2,2,2) : D[dyi,prmj]
        sdp = dp.shape # (m,n/2,2,2)
        # The common dimensions are m = number of samples and 2, number space dimensions.
        dps = np.transpose(dp,(1,3,0,2)).reshape(sdp[1],sdp[3],sdp[0]*sdp[2]) # (n/2,2,2*m) : D[dyi,prmj]
        ds2 = 2. * np.dot(dps, dpm.flatten() * lwf).flatten() # / np.sum(lwf) # (n,) : D[s,prmj] = sum_i=1,m[ 2 * wi * dyi * D[dyi,prmj] ]
        return ds2


    def fit_x0_lcoeff(self, xdata, ydata, ysigm=np.array([]),
                      x0=np.array([0.,0.]), lcoeff0=np.array([]),
                      luse=np.array([])):
        '''
        Fits the projection to a set of data including an x0 shift
        instead of the shift coefficients (order 0) using a minimum
        least-squares approach.

        Parameters:
            xdata : array_like, shape=(m,2)
                object plane position data
            ydata : array_like, shape=(m,2)
                image plane position data
            ysigm : array_like, shape=(m,2), optional
                image plane position error
            x0 : tuple, optional
                x data shift, initializes to [0,0] of not given
            lcoeff : array_like, float, optional
                sequence of projection coefficient interpreted as tuples {[alk0,alk1]}
                initializes to the internal coefficient list of the object if not given
            luse : array_like, int, optional
                polynomial term flags

        Return:
            [prm, cov, msd]
            prm : numpy.ndarray shape(num_terms * 2)
                fitted parameters with (y) shift coefficient at index 0 replaced by the x0 shift
                sorted [x00,x01, ... , alk0, alk1, ...]
            cov : numpy.ndarray shape(num_terms * 2, num_terms * 2)
                covariance matrix, entries are sorted as is prm
            msd : float
                mean square deviation between data and model per data item

        '''
        #
        lcbk = self.lcoeff #  backup coefficients
        lubk = self.luse # backup use flags
        if np.size(lcoeff0) > 0: self.lcoeff = lcoeff0
        if np.size(luse) > 0: self.luse = luse
        self.set_coeff_idx(0, 0., 0.) # no shift
        self.set_use_idx(0, 0) # no shift
        # n = self.num_terms
        # prm0 = np.zeros(2*n, dtype=float)
        # if np.size(x0)>=2: prm0[0:2] = x0[0:2]
        # prm0[2:n] = self.lcoeff.flatten()[2:n] # copy initial parameters (except shift)
        prm0 = self.__getprm_fitx0(x0, self.lcoeff)
        # data
        m2 = min(np.size(xdata),np.size(ydata))
        m = int((m2 - m2%2) / 2) # number of data tuples
        # store data lists as class members
        self.__xdata_fitx0 = np.array(xdata).flatten()[0:2*m].reshape(m,2) # x
        self.__ydata_fitx0 = np.array(ydata).flatten()[0:2*m].reshape(m,2) # y
        self.__wdata_fitx0 = np.full((m,2), 1.) # weights
        # setup weights from y error estimates: wi = 1/sigmai**2
        if np.size(ysigm) >= m2:
            ys = np.array(ysigm).reshape(m,2)
            yst = np.transpose(ys)
            wst = np.array([self.__getweights(yst[0]**2), self.__getweights(yst[1]**2)])
            self.__wdata_fitx0 = np.transpose(wst)
        # minimize weighted square deviations
        nmsol = minimize(self.__sdevproj_x0, prm0, method='BFGS', jac=self.__dsdevproj_x0)
        prmf = nmsol.x
        sqrdev = self.__sdevproj_x0(prmf)
        prmcov = nmsol.hess_inv * sqrdev # np.sum(self.__wdata_fitx0.flatten()) #  rethink this scaling
        # resort result vectors for output
        prml = np.zeros(2 * self.num_terms)
        covl = np.zeros((2 * self.num_terms, 2 * self.num_terms))
        lj = 0
        for j in range(0, self.num_terms): # covl and prml rows
            if (j > 0 and self.luse[j] == 0): continue # skip unused coefficient
            j2 = 2 * j
            lj2 = 2 * lj
            prml[j2:j2+2] = prmf[lj2:lj2+2] # copy prm
            li = 0
            for i in range(0, self.num_terms): # covl columns
                if (i > 0 and self.luse[i] == 0): continue # skip unused coefficient
                i2 = 2 * i
                li2 = 2 * li
                covl[j2:j2+2,i2:i2+2] = prmcov[lj2:lj2+2,li2:li2+2] # copy cov
                li = li + 1
            lj = lj + 1
        # restore backup
        self.lcoeff = lcbk # coefficients
        self.luse = lubk # use flags
        # return fit result
        return [prml, covl, sqrdev]


    # lcoeff[:] -> prm
    def __getlcoeff_lstsq(self, lprm):
        '''
        Sort a parameter array obtained by linear least square parameter fit
        with method fit_lcoeff to internal lcoeff order.

        This method is private and should only be called from fit_lcoeff.
        
        Parameters:
            lprm : numpy.ndarray (n,)
                a list of coefficients as used by the least squares fitting
        
        Return:
            numpy.ndarray, shape (self.num_terms,2)
                a polynomial coefficient list ordered as the internal lcoeff member

        '''
        # all inputs are expected as numpy.ndarray
        luse = self.__luse_term.copy()
        lc = np.zeros((self.num_terms, 2))
        l = 0
        for i in range(0,self.num_terms):
            if luse[i] == 1:
                l2 = 2 * l
                lc[i,0:2] = lprm[l2:l2+2]
                l = l + 1
        return lc

    def __getmat_lstsq(self, xdata, weights=np.array([])):
        '''
        Calculates the derivatives of the polynomial with respect to the
        polynomial coeffcients for given input data and sorts them
        into a matrix for the linear least squares method.

        This method is private and should only be called from fit_lcoeff.

        Parameters:
            xdata : numpy.ndarray shape(m,2)
                list of x data tuples
            weights : numpy.ndarray shape(m,2) optional
                list of weights for each data row
                if weights is not given, or not of the same shape as xdata,
                the errors default to 1

        Return:
            numpy.ndarray shape(2*m, n)
                matrix of the linear equation system to be solved
        '''
        # input is expected as numpy.ndarray
        m = xdata.shape[0]
        lw = np.full(2*m, 1., dtype=float)
        if (xdata.shape == weights.shape):
            lw = weights.flatten()
        ldc = self.project_deriv_lcoeff(xdata) # (m,num_terms,2,2) coefficient derivatives at xdata
        luse = self.luse
        n = int(2 * np.sum(luse)) # number of coefficients
        ldp = np.zeros((2*m,n), dtype=float) # prepare result array
        for i in range (0, m): # loop data rows
            i2 = 2 * i
            l = 0
            for j in range(0, self.num_terms): # loop coefficient columns
                if (luse[j]==0): continue # skip not flagged terms
                l2 = 2 * l
                dcij = ldc[i,j].T # transpose of 2 x 2 coeffcient derivate submatrix
                                  # [[dyi0/daj0, dyi0/daj1],[dyi1/daj0, dyi1/daj1]]
                #print([i,j,l,ldp[i2,l2:l2+2],dcij[0] * lw[i2]])
                ldp[i2,l2:l2+2] = dcij[0] * lw[i2] # copy the weighted  2 x 2 coeffcient derivate submatrix
                #print([i,j,l,ldp[i2+1,l2:l2+2],dcij[1] * lw[i2+1]])
                ldp[i2+1,l2:l2+2] = dcij[1] * lw[i2+1] # copy the weighted  2 x 2 coeffcient derivate submatrix
                l = l + 1
        return ldp

    def fit_lcoeff(self, xdata, ydata, ysigm=np.array([]), luse=np.array([])):
        '''
        Fits the projection to a set of data using a linear least-squares.

        Parameters:
            xdata : array_like, shape=(m,2)
                object plane position data
            ydata : array_like, shape=(m,2)
                image plane position data
            ysigm : array_like, shape=(m,2), optional
                image plane position error
            luse : array_like, int, optional
                polynomial term flags

        Return:
            [prm, cov, msd]
            prm : numpy.ndarray shape(num_terms * 2)
                fitted parameters sorted like the flatted coefficient list lcoeff
                [... ,alk0, alk1, ...]
            cov : numpy.ndarray shape(num_terms * 2, num_terms * 2)
                covariance matrix, entries are sorted as is prm
            msd : float
                mean square deviation between data and model per data item

        '''
        #
        lcbk = self.lcoeff #  backup coefficients
        lubk = self.luse # backup use flags
        if np.size(luse) > 0: self.luse = luse
        # data
        m2 = min(np.size(xdata),np.size(ydata))
        m = int((m2 - m2%2) / 2) # number of data tuples
        # store data lists as private class members
        self.__xdata_fit = np.array(xdata).flatten()[0:2*m].reshape(m,2) # x
        self.__ydata_fit = np.array(ydata).flatten()[0:2*m].reshape(m,2) # y
        self.__wdata_fit = np.full((m,2), 1.) # weights
        # setup weights from y error estimates
        if (np.size(ysigm) >= m2):
            ys = np.array(ysigm).reshape(m,2)
            yst = np.transpose(ys)
            wst = np.array([self.__getweights(yst[0]), self.__getweights(yst[1])])
            self.__wdata_fit = np.transpose(wst)
        # minimize weighted square deviations
        mdesign = self.__getmat_lstsq(self.__xdata_fit, self.__wdata_fit) # (2*m,2*num_terms)
        # print(mdesign)
        vbeta = self.__ydata_fit.reshape(2*m) * self.__wdata_fit.reshape(2*m) # (2*m)
        # print(vbeta)
        #####
        # use a linear least square solver here !
        # but we want one which returns the covariance matrix of the solution as well !
        # this is possible (see numerical recipes chapter 15)
        # but seems not implemented in any of the numpy or scipy codes
        # therefore emilys implements this in emilys.numerics.lstsq.linear_lstsq
        #####
        lsol = llstsq(mdesign, vbeta)
        prmf = lsol.x[0]
        prmcov = lsol.cov
        sqrdev = lsol.chisq[0]
        # resort result vectors for output
        #prml = np.zeros(2 * self.num_terms)
        prml = self.__getlcoeff_lstsq(prmf).flatten()
        covl = np.zeros((2 * self.num_terms, 2 * self.num_terms))
        lj = 0
        for j in range(0, self.num_terms): # covl and prml rows
            if (self.luse[j] == 0): continue # skip unused coefficient
            j2 = 2 * j
            lj2 = 2 * lj
            covl[j2] = self.__getlcoeff_lstsq(prmcov[lj2]).flatten()
            covl[j2+1] = self.__getlcoeff_lstsq(prmcov[lj2+1]).flatten()
            lj = lj + 1
        # restore backup
        self.lcoeff = lcbk # coefficients
        self.luse = lubk # use flags
        # return fit result
        return [prml, covl, sqrdev]