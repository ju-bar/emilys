# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:51:00 2023
@author: ju-bar

Class implementation calculating inelastic scattering coefficients
for phonon EELS using a density of states

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

import numpy as np
from numba import jit, int64, float64, complex128
from scipy import interpolate
import emilys.optics.econst as ec
import emilys.optics.tp1dho as ho
import emilys.optics.waki as sfac

def pbolz(e, t):
    '''
    
    Returns values of the un-normalized boltzmann distribution

    Parameters
    ----------
        e : float
            energy in eV
        t : float
            temperature in K

    Returns
    -------
        float
            value of the boltzmann distribution
    
    '''
    return np.exp(-e*ec.PHYS_QEL/(ec.PHYS_KB*t))

def nmaxt(e, t, pthr):
    '''
    
    Returns the maximum quantum number to be below
    a relative threshold pthr at temperature t for
    a given oscillator energy ep

    Parameters
    ----------

        e : float
            oscillator energy in eV
        t : float
            temperature in K
        pthr : float
            probability threshold (< 1)

    Returns
    -------
        int
            first quantum oscillator level with an
            occupation below the threshold compared
            to the ground state occupation

    '''
    p0 = pbolz(0.5*e, t)
    pt = pthr * p0
    n = 1 - int(0.5 + ec.PHYS_KB*t * np.log(pt) / (e*ec.PHYS_QEL))
    return n

@jit(float64(float64, float64[:], float64[:]), nopython=True)
def get_fe(q, q0, f0):
    '''

    get_fe

    Linear interpolation kernel for scattering factors.
    Compiled code.

    Parameters
    ----------
        q : float
            q-vector length in 1/A
        q0 : array of floats
            q grid from which to interpolate
        f0 : array of floats
            data that is the basis for the interpolation

    Returns
    -------
        float
            interpolated scattering factor
    
    '''
    f = q / q0[1]
    i0 = int(f)
    fi = f - i0
    return (f0[i0] * (1.0 - fi) + f0[i0+1] * fi)

def get_fen(q, q0, f0):
    '''

    get_fen

    Linear interpolation kernel for scattering factors.

    Parameters
    ----------
        q : array of floats
            q-vector length in 1/A
        q0 : array of floats
            q grid from which to interpolate
        f0 : array of floats
            data that is the basis for the interpolation

    Returns
    -------
        float
            interpolated scattering factor
    
    '''
    n = len(q)
    feq = np.zeros(n, dtype=np.float32)
    for i in range(0, n):
        feq[i] = get_fe(q[i], q0, f0)
    return feq

def numint_pdos(l_ev, l_pdos):
    '''

    Calculates the numerical integral of given PDOS data.
    The data is expected in two arrays of equal length.
    The data will be sorted, so unsorted input is accepted.

    If not specifically given, a zero value is added
    for zero energy. Another zero value is appended to
    terminate the PDOS. The modified and sorted PDOS data
    is also returned

    Parameters
    ----------
        l_ev : array-like, type float
            energy values
        l_pdos : array-like, type float
            pdos values

    Returns
    -------
        float
            integral of the data
        numpy.array, type float
            sorted and extended energy grid used
        numpy.array, type float
            pdos data corresponding to the energy grid

    '''
    e0 = np.array(l_ev) # make a copy of the input energy array
    ne0 = len(e0)
    ne1 = ne0 + 1
    assert len(l_pdos)==ne0, "Expecting equal length of input arrays."
    e_min = np.amin(l_ev)
    e_max = np.amax(l_ev)
    e_cap = 2 * e_max
    i1 = 0
    if e_min > 0.0:
        ne1 += 1
        i1 = 1
    e1 = np.zeros(ne1, dtype=float)
    p1 = e1 * 0.0
    for i in range(0, ne0):
        j = np.argmin(e0) # find smallest energy
        e1[i1] = e0[j] # transfer energy to sorted energy list
        p1[i1] = l_pdos[j] # transfer pdos value
        e0[j] = e_cap # invalidate the transfert energy
        i1 += 1 # advance in sorted list
    # integral
    s = np.trapz(p1[0:i1], x=e1[0:i1])
    return s, e1[0:i1], p1[0:i1]


def resample_pdos(l_ev, l_pdos, dE, pdos_ip_kind='linear'):
    '''

    Resamples the input PDOS to a new equidistant energy grid
    of step size dE.

    The input data is expected in two arrays of equal length.
    The data will be sorted, so unsorted input is accepted.
    
    This function preserves the norm of the pdos and returns
    the integral value.

    Parameters
    ----------
        l_ev : array-like, type float
            energy values
        l_pdos : array-like, type float
            pdos values
        dE : float
            output sampling rate of energies
        pdos_ip_kind : str, default: 'linear'
            interpolation kind, see documentation of scipy.interpolate.interp1d

    Returns
    -------
        float
            integral of the pdos data
        numpy.array, type float
            new energy grid
        numpy.array, type float
            new pdos data
    
    '''
    s1, e1, p1 = numint_pdos(l_ev, l_pdos)
    ip1 = interpolate.interp1d(e1, p1, kind=pdos_ip_kind,copy=True,bounds_error=False,fill_value=0.0)
    e2 = np.arange(0., np.amax(e1) + dE, dE)
    p2 = ip1(e2)
    s2 = np.trapz(y=p2, x=e2)
    p2n = p2 * s1 / s2 # keep input norm
    return s1, e2, p2n

# would be nice to have compiled, this costs a lot and is repeated a lot
@jit(int64(int64, int64[:], int64, int64, int64, int64[:,:], complex128[:], complex128[:], float64[:,:], float64[:,:]), nopython=True)
def numint_muh(n, l_qi, iqex, iqey, nd, dethash, tmx, tmy, feq, muh):
    '''
    numint_muh

    Calculates mu_h for a given pair of matrix elements hx, hy, detector
    function dethash and electron scattering factors feq into seq.

    Compiled code using jit.
    int64(int64, int64[:], int64, int64, int64, int64[:,:], complex128[:], complex128[:], float64[:,:], float64[:,:])

    Parameters
    ----------
        n : int64
            number of pixels of the original grid
        l_qi : int64[:], shape=(n)
            frequency indices on the original grid
        iqex : int64
            offset of the original grid in the extended grid x dimension
        iqey : int64
            offset of the original grid in the the extended grid y dimension
        nd : int64
            number of detector pixels
        dethash : int64[:,:], shape=(nd, 2)
            pixel indices on the original q-grid falling into the detector
        tmx : complex128[:], shape(nextended_x)
            transition matrix element of the x mode, calculated on the extended grid
        tmy : complex128[:], shape(nextended_y)
            transition matrix element of the y mode, calculated on the extended grid
        feq: float64[:,:], shape(nextended_y,nextended_x)
            electron scattering factor on the extended grid
        muh: float64[:,:]
            resulting mu_{h,0} for h frequencies of the original grid

    Returns
    -------
        int64
            always = 0

    '''
    for i2 in range(0, n): # run over indices of the original grid -> hy
        #j2 = l_qi[i2] # frequency index of the output grid (hy)
        for i1 in range(0, n): # run over indices of the original grid -> hx
            #j1 = l_qi[i1] # frequency index of the output grid (hx)
            s = 0.0 # reset accumulator
            for idet in range(0, nd): # run over detector pixels -> (qy, qx)
                jdet = dethash[idet] # (qy, qx) indices in the original grid
                jq2 = jdet[0] + iqey # index of qy in the extended array
                jq1 = jdet[1] + iqex # index of qx in the extended array
                jqh2 = l_qi[jdet[0]] + i2 + iqey # index of (qy + hy) in the extended array
                jqh1 = l_qi[jdet[1]] + i1 + iqex # index of (qx + hx) in the extended array
                pqh = tmx[jqh1] * tmy[jqh2] # tmx(qx + hx) * tmy(qy + hy)
                pq = tmx[jq1] * tmy[jq2] # tmx(qx) * tmy(qy)
                p = np.double((pq * np.conjugate(pqh)).real) # product of 2d matrix elements -> is always real valued
                trs = p * feq[jqh2,jqh1] * feq[jq2, jq1] # multiplication with electron scattering factors
                s += trs # summation over all pixels (qy, qx) in the detector
            muh[i2, i1] = s # store result in output
    return 0

@jit(int64(int64[:], int64, int64, int64[:,:], complex128[:], complex128[:], float64[:,:], float64[:,:]), nopython=True)
def numint_muh2(l_qi, iqex, iqey, dethash, tmx, tmy, feq, muh):
    '''
    numint_muh2

    Calculates mu_h for a given pair of matrix elements hx, hy, detector
    function dethash and electron scattering factors feq into seq.

    Compiled code using jit.
    int64(int64[:], int64, int64, int64[:,:], complex128[:], complex128[:], float64[:,:], float64[:,:])

    Parameters
    ----------
        l_qi : int64[:], shape=(n)
            frequency indices on the original grid
        iqex : int64
            offset of the original grid in the extended grid x dimension
        iqey : int64
            offset of the original grid in the the extended grid y dimension
        dethash : int64[:,:], shape=(nd, 2)
            pixel indices on the original q-grid falling into the detector
        tmx : complex128[:], shape(nextended_x)
            transition matrix element of the x mode, calculated on the extended grid
        tmy : complex128[:], shape(nextended_y)
            transition matrix element of the y mode, calculated on the extended grid
        feq: float64[:,:], shape(nextended_y,nextended_x)
            electron scattering factor on the extended grid
        muh: float64[:,:]
            resulting mu_{h,0} for h frequencies of the original grid

    Returns
    -------
        int64
            always = 0

    '''
    n = len(l_qi)
    n2 = n >> 1 # q-grid nyquist
    nd = len(dethash)
    for idet in range(0, nd): # run over detector pixels -> (qy, qx)
        jdet = dethash[idet] # (qy, qx) indices in the original grid
        jq2 = jdet[0] + iqey # index of qy in the extended array
        jq1 = jdet[1] + iqex # index of qx in the extended array
        pq = tmx[jq1] * tmy[jq2] # tmx(qx) * tmy(qy)
        feqq = feq[jq2, jq1] # fe(q)
        #
        jqh20 = jq2 - n2 # offset of the qy + hy grid in the extended grid
        jqh10 = jq1 - n2 # offset of the qx + hx grid in the extended grid
        #
        # run through the h vectors and sum to output
        for i2 in range(0, n): # hy
            j2 = i2 + jqh20
            for i1 in range(0, n): # hx
                j1 = i1 + jqh10
                pqh = tmy[j2] * tmx[j1] # matrix elements for q + h
                p = np.double((pq * np.conjugate(pqh)).real) # product of 2d matrix elements -> is always real valued
                muh[i2,i1] += p * feq[j2,j1] * feqq
        #
    return 0


# ----------------------------------------------------------------------
#
# CLASS phonon isc
#
# ----------------------------------------------------------------------

class phonon_isc:
    '''

    class phonon_isc

    Attributes
    ----------
        sfwa : emilys.optics.waki.waki
            instance of Waasmaier and Kirfel electron scattering factors
        t : float
            temperature in K
        tev : float
            thermal energy in eV
        pthr : float
            probability threshold for limiting sums
        qgrid : dict
            q-space grid data
        det : dict
            detector data
        pdos : dict
            phonon density of states data
        atom : dict
            atom data
        l_dE : numpy.ndarray, dtype=float
            energy loss grid used in the last calculation in eV

    Members
    -------
        set_qgrid : Set q-grid parameters for numerical calculation (square size)
        set_atom : Set atom data and prepares scattering factors
        set_detector : Set detector data and prepare related functions
        set_temperatur : Set the temperature to assume in calculations
        set_pdos : Set a phonon density of states (PDOS)

    Usage
    -----
        1) init by e.g. pmu = phonon_isc()
        2) pmu.set_qgrid(qmax, n)
        3) pmu.set_detector(0., 1., [0., 0.])
        4) pmu.prepare_qgrid_ext()
        5) pmu.set_atom(14, "Si", 28.0855)
        6) pmu.set_pdos(l_e, l_p, 0.005)
        7) pmu.set_temperature(293.0)
        8) a = pmu.get_mul2d([0.001, 0.01]) # ... to calculate

    '''
    def __init__(self, prob_threshold=0.05):
        self.sfwa = sfac.waki() # electron scattering factors
        self.qgrid = { # q-space grid data
            "qmax" : 12, # max. diffraction vector in 1/A
            "n" : 600 # number of samples
        }
        self.qgrid["dq"] = 2 * self.qgrid["qmax"] / self.qgrid["n"] # sampling rate 1/A / pixel
        self.qgrid["nyquist"] = (self.qgrid["n"] >> 1) # nyquist number
        self.qgrid["l_qi"] = np.arange(-self.qgrid["nyquist"], -self.qgrid["nyquist"] + self.qgrid["n"], 1, dtype=np.int32)
        self.qgrid["l_q"] = (self.qgrid["l_qi"] * self.qgrid["dq"]).astype(np.double) # q grid
        self.t = 300. # temperature in K
        self.tev = self.t * ec.PHYS_KB / ec.PHYS_QEL # thermal energy in eV
        self.pthr = prob_threshold # probability weight threshold
        self.det = { # detector definition
            "inner" : 0.0,
            "outer" : 1.0,
            "center" : [0.0, 0.0]
        }
        self.pdos = { # phonon density of states
        }
        self.atom = { # atom definition
        }

    def set_qgrid(self, qmax, n):
        '''

        Set q-grid parameters for numerical calculation (square size)

        Set this before using other functions depending on the grid settings:
        detector, atom

        Parameters
        ----------
            qmax : float
                maximum diffraction vector along x and y in 1/A
            n : int
                number of samples to represent -qmax to qmax

        Returns
        -------
            None

        Remarks
        -------
            The sample at +qmax might be omitted depending on n,
            but the sample at -qmax is always present.
        
        '''
        self.qgrid["qmax"] = qmax # max. diffraction vector in 1/A
        self.qgrid["n"] = n # number of samples
        self.qgrid["nyquist"] = (self.qgrid["n"] >> 1) # nyquist number
        self.qgrid["dq"] = 2 * self.qgrid["qmax"] / self.qgrid["n"] # sampling rate 1/A / pixel
        self.qgrid["l_qi"] = np.arange(-self.qgrid["nyquist"], -self.qgrid["nyquist"] + self.qgrid["n"], 1, dtype=np.int64)
        self.qgrid["l_q"] = (self.qgrid["l_qi"] * self.qgrid["dq"]).astype(np.double) # q grid

    def prepare_atom(self):
        '''

        prepare_atom

        Called by the object to prepare scattering factors.

        Returns
        -------
            None
        
        '''
        n = 20 * self.qgrid["n"]
        self.atom["data"] = {}
        adat = self.atom["data"] # setup the fine q grid for atomic scattering factors
        adat["n"] = n
        adat["dq"] = 2 * self.qgrid["qmax"] / adat["n"] # sampling rate 1/A / pixel
        adat["q"] = (np.arange(0, n+1, 1) * adat["dq"]).astype(np.double)
        adat["fe"] = (adat["q"] * 0.0).astype(np.double)
        for i in range(0, n): # prepare the scattering factor on a fine q grid
            adat["fe"][i] = np.double(self.sfwa.get_fe(self.atom["Z"], 0.5 * adat["q"][i]))


    def set_atom(self, Z, symbol, mass):
        '''

        set_atom

        Set atom data in the object and prepare scattering factors.

        Requires prior call of members
        set_qgrid

        Parameters
        ----------
            Z : int
                atomic number
            symbol : str
                atom symbol
            mass : float
                mass of the atom in atomic mass units (Dalton)

        Returns
        -------
            None
        
        '''
        self.atom["Z"] = int(Z)
        self.atom["symbol"] = str(symbol)
        self.atom["mass"] = mass
        self.prepare_atom()
        get_fe(0.0, self.atom["data"]["q"], self.atom["data"]["fe"]) # call the interpolation routine to compile it

    def prepare_detector(self):
        '''

        prepare_detector

        Called by the object to prepare the detector functions.
        Writes data in the attribute det.

        
        Returns
        -------
            None
        
        '''
        nq = self.qgrid["n"]
        l_q = self.qgrid["l_q"]
        l_qi = self.qgrid["l_qi"]
        l_det = np.zeros((nq,nq), dtype=np.double) # detector function
        l_dethash = []
        q0 = self.det["center"]
        qd0 = self.det["inner"]
        qd1 = self.det["outer"]
        l_dqy2 = (l_q - q0[0])**2
        l_dqx2 = (l_q - q0[1])**2
        iqr = [[nq,-nq],[nq,-nq]] # min and max frequencies of the detector in rows and columns
        for i in range(0, nq): # qy
            for j in range(0, nq): # qx
                q = np.sqrt(l_dqy2[i] + l_dqx2[j])
                if (q >= qd0) and (q < qd1):
                    l_det[i,j] = 1.
                    l_dethash.append([i,j])
                    iqr[0][0] = min(iqr[0][0], l_qi[i])
                    iqr[0][1] = max(iqr[0][1], l_qi[i])
                    iqr[1][0] = min(iqr[1][0], l_qi[j])
                    iqr[1][1] = max(iqr[1][1], l_qi[j])
        self.det["grid"] = l_det
        self.det["hash"] = {
            "index" : np.array(l_dethash, dtype=np.int64),
            "ifreq_range" : np.array(iqr, dtype=np.int64)
        }

    def set_detector(self, inner, outer, center):
        '''

        set_detector

        Set detector data and prepare related functions.

        Requires prior calls of members
        set_qgrid

        Parameters
        ----------
            inner : float
                inner collection range limitation in 1/A
            outer : float
                outer collection range limitation in 1/A
            center : array of floats (len = 2)
                center position (qy, qx) of the detector in the diffraction plane in 1/A
        
        Returns
        -------
            None
        
        '''
        self.det["inner"] = inner
        self.det["outer"] = outer
        self.det["center"] = center
        self.prepare_detector()

    def set_temperatur(self, T):
        '''

        set_temperatur

        Set the temperature to assume in calculations.

        Parameters
        ----------
            T : float
                temperature in K

        Returns
        -------
            None

        '''
        self.t = T
        self.tev = self.t * ec.PHYS_KB / ec.PHYS_QEL

    def set_pdos(self, l_ev, l_pdos, dE, pdos_ip_kind='linear', sub_sample=0.01 ):
        '''

        set_pdos 

        Set a phonon density of states (PDOS) from a list
        of energy values (l_ev) and a list of pdos values (l_pdos).

        Generates an internal list of probabilities on a new energy
        grid with step size dE starting from dE and covering the
        range in l_ev. Assumes sorted input (l_ev) with ascending
        energy and properly normalized l_pdos.

        The energy dE is the step size of phonon energies used for
        calculating the inelastic scattering factors. It is also the
        step size used to output energy losses. This parameter
        strongly determines the calculation time.

        Results are stored in the object property pdos.

        Parameters
        ----------
            l_ev : numpy.ndarray of floats
                List of energy samples in eV for each sample of the pdos
            l_pdos : numpy.ndarray of floats
                List of PDOS samples for each sample of energy
            dE : float
                phonon energy step size in eV
            pdos_ip_kind : str, default: 'linear'
                interpolation kind, see documentation of scipy.interpolate.interp1d
            sub_sample : float, default: 0.01
                sub-sampling of the energy grid for bin integration

        Returns
        -------
            None

        '''
        assert len(l_ev) == len(l_pdos), "Expecting equal length of lists l_ev and l_pdos."
        assert dE > 0.001, "Expecting non-negative energy step size dE."
        # backup input pdos
        self.pdos["original"] = {
            "energy" : l_ev,
            "pdos" : l_pdos
        }
        # resample the pdos on a fine energy grid
        dEfine = sub_sample * dE
        s1, x1, y1 = resample_pdos(l_ev, l_pdos, dEfine, pdos_ip_kind=pdos_ip_kind)
        self.pdos["original"]["norm"] = s1
        # prepare bins of 
        x2 = np.arange(0., np.amax(l_ev)+dE, dE) # target phonon energy bins
        y2 = x2 * 0.0 # integrated pdos
        nep = len(x2) # number of bins
        # integrate probability in bins
        for i in range(1, nep-1): # loop over bins and integrate probabilities
            ei = [(x2[i] + x2[i-1]) * 0.5, (x2[i+1] + x2[i]) * 0.5] # energy bin interval
            ii = [int(ei[0] / dEfine), int(ei[1] / dEfine)] # reference pixels on fine grid
            si = np.trapz(y1[ii[0]:ii[1]+1],x=x1[ii[0]:ii[1]+1])
            y2[i] = si
        # store pdos data
        self.pdos["data"] = {
            "energy_step" : dE,
            "energy" : x2[1:nep-1],
            "pdos" : y2[1:nep-1],
            "ip_kind" : pdos_ip_kind
        }

    def prepare_qgrid_ext(self):
        '''

        prepare_qgrid_ext

        Prepares an extended q-grid for the mu-integration.

        Requires prior calls to members
        set_qgrid, set_detector

        The size of the grid depends on the original q grid
        and on the detector (on its size and position).
        It is extended compared to the primary q-grid by the
        range of frequencies on the grid that is covered by
        the detector.

        This function is called from class functions and
        should not be used elsewhere. It relies on a full
        setup of the object (grid and detector)

        '''
        l_qi = self.qgrid["l_qi"] # list of q frequency indices
        l_q_rng = [np.amin(l_qi), np.amax(l_qi)]
        l_det_idr = self.det["hash"]["ifreq_range"] # frequency range of the detector on the q-grid
        l_ext_rng = np.array([l_q_rng, l_q_rng]) # preset extended range by primary range
        l_ext_rng[0,0] = min(l_ext_rng[0,0], l_ext_rng[0,0] + l_det_idr[0,0]) # update minimum ifreq of y
        l_ext_rng[0,1] = max(l_ext_rng[0,1], l_ext_rng[0,1] + l_det_idr[0,1]) # update maximum ifreq of y
        l_ext_rng[1,0] = min(l_ext_rng[1,0], l_ext_rng[1,0] + l_det_idr[1,0]) # update minimum ifreq of x
        l_ext_rng[1,1] = max(l_ext_rng[1,1], l_ext_rng[1,1] + l_det_idr[1,1]) # update maximum ifreq of x
        l_qiy = np.arange(l_ext_rng[0,0], l_ext_rng[0,1] + 1, 1, dtype=np.int64)
        l_qix = np.arange(l_ext_rng[1,0], l_ext_rng[1,1] + 1, 1, dtype=np.int64)
        # store extended grid data
        self.qgrid["extended"] = {
            "ny" : len(l_qiy), # number of extended columns
            "nx" : len(l_qix), # number of extended rows
            "ifreq_range" : l_ext_rng, # extended frequency range [iqy,iqx]
            "l_qiy" : l_qiy, # y frequency indices
            "l_qix" : l_qix, # x frequency indices
            "offset" : np.array([l_qi[0] - l_qiy[0],l_qi[0] - l_qix[0]],dtype=np.int64) # offset of the original q-grid in the extended grid (y,x)
        }

    def prepare_feq_ext(self):
        '''

        prepare_feq_ext

        Prepares atom scattering factors on an extended q-grid
        for the mu-integration.

        This function is called from class functions and
        should not be used elsewhere. It relies on a full
        setup of the object (grid and detector)

        '''
        gex = self.qgrid["extended"]
        ny = gex["ny"]
        nx = gex["nx"]
        dq = self.qgrid["dq"]
        l_q2ex = np.sum(np.meshgrid((gex["l_qix"] * dq)**2, (gex["l_qiy"] * dq)**2), axis=0)
        gex["l_qabs"] = np.sqrt(l_q2ex)
        gex["l_feq"] = get_fen(gex["l_qabs"].flatten(), self.atom["data"]["q"], self.atom["data"]["fe"]).reshape(ny,nx).astype(np.double)

    def get_trstr(self, ep, mx, nx, my, ny):
        '''

        get_trstr

        Calculates the transition strength for all points
        of the q grid using the extended grid.

        This function is called from class functions and
        should not be used elsewhere. It relies on a full
        setup of the object (grid and detector)

        '''
        n = self.qgrid["n"]
        dq = self.qgrid["dq"]
        l_qi = self.qgrid["l_qi"]
        pfac = dq * dq * (ec.PHYS_HPL**2 / (2*np.pi * ec.EL_M0))**2 # constant prefactor (h^2 / (2 pi m_el))^2 dqx dqy
        gex = self.qgrid["extended"]
        l_qex = gex["l_qix"] * dq
        l_qey = gex["l_qiy"] * dq
        feq = gex["l_feq"]
        l_gsh = gex["offset"]
        # get oscillator parameters
        w = ep / ec.PHYS_HBAREV # frequency in Hz
        mat = self.atom["mass"] * ec.PHYS_MASSU # atom mass in kg
        u02 = ho.usqr0(mat, w) * 1.E20 # ground state MSD in A^2
        hx = ho.tsq(l_qex, u02, mx, nx) # x mode transition factors <a_nx|H|a_mx>(q)
        hy = ho.tsq(l_qey, u02, my, ny) # y mode transition factors <a_ny|H|a_my>(q)
        sdet = np.zeros((n,n), dtype=np.double) # detector sum on the original q grid
        dethash = self.det["hash"]["index"]
        ndet = len(dethash)
        #j = numint_muh(n, l_qi, l_gsh[1], l_gsh[0], ndet, dethash, hx, hy, feq, sdet)
        j = numint_muh2(l_qi, l_gsh[1], l_gsh[0], dethash, hx, hy, feq, sdet)
        # for i2 in range(0, n):
        #     j2 = l_qi[i2] # frequency index of the output grid (hy)
        #     for i1 in range(0, n):
        #         j1 = l_qi[i1] # frequency index of the output grid (hx)
        #         s = 0.0 # reset accumulator
        #         for idet in range(0, len(dethash)):
        #             jdet = dethash[idet]
        #             jd2 = l_qi[jdet[0]] # frequency index of the detector grid (qy)
        #             jd1 = l_qi[jdet[1]] # frequency index of the detector grid (qx)
        #             ld2 = jd2 - l_ifr[0,0] # index of qy in the extended array
        #             ld1 = jd1 - l_ifr[1,0] # index of qx in the extended array
        #             k2 = j2 + jd2 # frequency index of hy + qy
        #             k1 = j1 + jd1 # frequency index of hx + qx
        #             l2 = k2 - l_ifr[0,0] # index of hy + qy in the extended array
        #             l1 = k1 - l_ifr[1,0] # index of hx + qx in the extended array
        #             trs = hx[ld1] * hy[ld2] * np.conjugate(hx[l1] * hy[l2]) * feq[l2,l1] * feq[ld2, ld1]
        #             s += trs
        #         sdet[i2, i1] = s 
        return sdet * pfac



    def get_mul2d(self, energy_loss_range, single_phonon=False, verbose=0):
        '''

        get_mul2d

        Calculates inelastic scattering coefficients mu(dE,qy,qx)

        The calculation is performed for the current atom, detector,
        PDOS, temperature, and q-grid in a local approximation.
        
        The output will be a 3-dimensional array of coefficients,
        where the first dimension samples the requested energy-loss
        range in multiple steps of the internal PDOS sampling set
        with the method set_pdos. The second and third dimensions
        are the q-grid (qy, qx) defined by set_qgrid sorted according
        to the numpy.fft.fftfreq method.

        Parameters
        ----------
            energy_loss_range : array-like, len=2, type float
                lower and upper bound of energy-losses in eV
            single_phonon : boolean, default: False
                flag: limits to single-phonon excitations
            verbose : int, default: 0
                verbosity level for text output
            
        Returns
        -------
            dict
                "data" : numpy.ndarray, num_dim=3, dtype=float
                    inelastic scattering coefficients
                "l_dE" : numpy.ndarray, num_dim=1, dtype=float
                    energy-loss grid in eV
                "l_q" : numpy.ndarray, num_dim=1, dtype=float
                    q-grid in 1/A along the second and third dimension of "data"

        '''
        assert energy_loss_range[0] < energy_loss_range[1], "Invalid energy loss range"
        # setup the energy loss grid
        dE = self.pdos["data"]["energy_step"] # set energy step from set_pdos
        pdos_thr = self.pthr * np.amax(self.pdos["data"]["pdos"])
        imin = int(energy_loss_range[0] / dE)
        imax = int(energy_loss_range[1] / dE)
        l_dE = np.arange(imin, imax+1, 1) * dE # energy loss grid
        a_mu = np.zeros((len(l_dE), self.qgrid["n"], self.qgrid["n"]), dtype=float) # output array
        # calculation
        # 1) Setup for given q-grid and detector
        #    This sets arrays used by other functions.
        #    We want to integrate of q in the detector (self.dethash)
        #    and this for offsets h spanning the complete q-grid.
        #    Because even in the local approximation, terms of q + h
        #    occur, this requires an extended q-grid. Grid points of
        #    the detector range must be shifted out to larger q for the
        #    integration.
        self.prepare_qgrid_ext() # prepares parameters of the extended q-grid
        if verbose > 0: print('(get_mul2d): extended grid freqency range:', self.qgrid["extended"]["ifreq_range"])
        #    Since the result is for the complete h grid, we calculate
        #    all factors for the extended q+h grid , which contains the
        #    q grid (in the case h=0). Factors are fe(q) and fe(q+h)
        #    for 2d q and h, also 1d matrix elements in x and y for
        #    each transition gamma_mx,nx(qx + hx) gamma_my,ny(qy + hy).
        if verbose > 0: print('(get_mul2d): calculating scattering factors on extended grid ...')
        self.prepare_feq_ext() # prepares fe(q) on the extended q-grid
        #
        # 2) Oscillators and states
        for iep in range(0, len(self.pdos["data"]["energy"])): # loop over phonon energies
            pdos = self.pdos["data"]["pdos"][iep] # pdos value in the current phonon energy bin
            if pdos < pdos_thr: # check pdos against probability threshold
                continue # skip phonon energy
            if verbose > 0: print('(get_mul2d): calculating contributions for phonon energy {:.4f} eV ...'.format(ep))
            ep = self.pdos["data"]["energy"][iep] # current phonon energy
            w = ep / ec.PHYS_HBAREV # oscillator frequency
            nimax = nmaxt(ep, self.t, self.pthr) # get max. initial state quantum number to take into account
            nrmtbd = (np.exp(ep/self.tev) - 1)**2 / np.exp(ep/self.tev) # normalization factor for the 2-d boltzmann distribution
            # boltzmann distribution threshold
            pb_thr = nrmtbd * pbolz(ep, self.t) * self.pthr # 2d ground state occupation time relative threshold
            # per phonon energy (to be weighted by pdos)
            for niy in range(0, nimax+1): # loop over initial states in the y dimension
                eiy = ho.En(niy, w) / ec.PHYS_QEL # initial y state energy in eV
                pby = pbolz(eiy, self.t) # initial y state Boltzmann distribution probability
                for nix in range(0, nimax+1): # loop over initial states in the x dimension
                    eix = ho.En(nix, w) / ec.PHYS_QEL # initial x state energy in eV
                    pbx = pbolz(eix, self.t) # initial x state Boltzmann distribution probability
                    pb2 = nrmtbd * pbx * pby # normalized occupation of the initial state
                    if pb2 < pbthr: continue # 
                    if verbose > 1: print('(get_mul2d): * initial state [{:d},{:d}] ...'.format(nix,niy))
                    #
                    # Initialize an expanding loop over final states.
                    # This loop will at least include single phonon excitations.
                    # It will terminate expanding x and y final states independently.
                    # The termination is made when during the current step of expansion
                    # only transitions smaller than a threshold self.pthr compared
                    # to the previous maximum was encountered.
                    #
                    nyd = 1 # y state phonon excitation level
                    nxd = 1 # x state phonon excitation level
                    ntr = 0 # count for included number of transitions
                    trsmax = 0. # records max transition strength
                    trsmaxx = 0. # records max transition strength in x states
                    trsmaxy = 0. # records max transition strength in y states
                    bmorefy = True # flags further expansion of y state transition levels
                    bmorefx = True # flags further expansion of x state transition levels
                    #
                    # loop over final states
                    #
                    while (bmorefx or bmorefy): # either add more columns or rows of higher multi-phonon levels by final states
                        #
                        if bmorefy: # add next rows (excluding corners)
                            trsmaxy = 0.
                            for nfy in [niy-nyd,niy+nyd]: # nfy row indices
                                if nfy < 0: continue # skip negative qn
                                efy = ho.En(nfy, w) / ec.PHYS_QEL # final y state energy
                                for nfx in range(nix-nxd+1,nix+nxd): # loop nfx from corner+1 to corner-1
                                    if nfx < 0: continue # skip negative qn
                                    if (niy == nfy) and (nix == nfx): continue # skip elastic (should actually not happen)
                                    if single_phonon and (abs(nfx-nix) + abs(nfy-niy) != 1): continue # skip multi-phonons in case of single-phonon calculation
                                    efx = ho.En(nfx, w) / ec.PHYS_QEL # final x state energy
                                    delE = efx - eix + efy - eiy # energy loss of the probing electron
                                    idelE = int(np.rint(delE / dE)) #
                                    if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                                        s = self.get_trstr(ep, nix, nfx, niy, nfy)
                                        trsmaxy = max(trsmaxy, np.amax(s))
                                        jdelE = idelE - imin
                                        a_mu[jdelE] += (pdos * pb2 * s)
                                        ntr += 1
                                    #
                            # exit criterion for final states y range
                            if (trsmax > 0.0) and (trsmaxy > 0.0): # global maximum present
                                if trsmaxy < 0.5 * self.pthr * trsmax: # max. on row loop is less than a threshold -> converged rows
                                    bmorefy = False
                        #
                        if bmorefx: # add columns (excluding corners)
                            trsmaxx = 0.
                            for nfx in [nix-nxd,nix+nxd]: # nfx outer column indices
                                if nfx < 0: continue # skip negative qn
                                efx = ho.En(nfx, w) / ec.PHYS_QEL # final x state energy
                                for nfy in range(niy-nyd+1,niy+nyd): # loop nfy from corner+1 to corner-1
                                    if nfy < 0: continue # skip negative qn
                                    if (niy == nfy) and (nix == nfx): continue # skip elastic (should actually not happen)
                                    if single_phonon and (abs(nfx-nix) + abs(nfy-niy) != 1): continue # skip multi-phonons in case of single-phonon calculation
                                    efy = ho.En(nfy, w) / ec.PHYS_QEL # final x state energy
                                    delE = efx - eix + efy - eiy # energy loss of the probing electron
                                    idelE = int(np.rint(delE / dE)) #
                                    if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                                        s = self.get_trstr(ep, nix, nfx, niy, nfy)
                                        trsmaxx = max(trsmaxx, np.amax(s))
                                        jdelE = idelE - imin
                                        a_mu[jdelE] += (pdos * pb2 * s)
                                        ntr += 1
                                    #
                            # exit criterion for final states x range
                            if (trsmax > 0.0) and (trsmaxx > 0.0): # global maximum present
                                if trsmaxx < 0.5 * self.pthr * trsmax: # max. on row loop is less than a threshold -> converged cols
                                    bmorefx = False
                        #
                        # handle corners
                        if not single_only: # corners are always multi-phonon excitations
                            # add corners (always needed as long as x or y progresses)
                            for nfy in [niy-nyd,niy+nyd]: # nfy corner row indices
                                if nfy < 0: continue # skip negative qn
                                efy = ho.En(nfy, w) / ec.PHYS_QEL # final y state energy
                                for nfx in [nix-nxd,nix+nxd]: # nfx corner columns indices
                                    if nfx < 0: continue # skip negative qn
                                    efx = ho.En(nfx, w) / ec.PHYS_QEL # final x state energy
                                    delE = efx - eix + efy - eiy # energy loss of the probing electron
                                    idelE = int(np.rint(delE / dE)) #
                                    if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                                        s = self.get_trstr(ep, nix, nfx, niy, nfy)
                                        trsmax = max(trsmax, np.amax(s))
                                        jdelE = idelE - imin
                                        a_mu[jdelE] += (pdos * pb2 * s)
                                        ntr += 1
                                    #
                        #
                        # update controls
                        if bmorefy:
                            nyd += 1 # add rows
                            trsmax = max(trsmax, trsmaxy) # maximum update
                        if bmorefx:
                            nxd += 1 # add columnss
                            trsmax = max(trsmax, trsmaxx) # maximum update
                        #
                        if single_only: # single-phonon transition case, just stop here
                            bmorefy = False
                            bmorefx = False
                        #
                    if verbose > 1: print('(get_mul2d): * transitions added: {:d}'.format(ntr))
        # returning
        return {
            "data" : a_mu,
            "l_dE" : l_dE,
            "l_q" : self.qgrid["l_q"]
        }

