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
from numba import jit, prange, int64, float64, complex128
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


@jit(int64(int64[:], int64, int64, int64[:,:], complex128[:], complex128[:], float64[:,:], float64[:,:]), nopython=True, parallel=True)
def numint_muh(l_qi, iqex, iqey, dethash, tmx, tmy, feq, muh):
    '''
    numint_muh

    Calculates mu_h for a given pair of matrix elements tmx, tmy, detector
    function dethash and electron scattering factors feq into muh.

    mu_{n,m}(h) = sum_q fe(h-q) fe(q) <a_n(t)|exp(-2pi I (h-q).t)|a_m(t)> <a_m(t)|exp(-2pi I q.t)|a_n(t)>

    This is vor vectors h, q, t, m, and n and the sum over q is performed for grid
    points in the detector collection aperture (given by dethash). Vector m is the
    initial state and n is the final state set of oscillator quantum numbers.

    Note also that
    <a_n(t)|exp(-2pi I q.t)|a_m(t)> = <a_m(t)|exp(-2pi I q.t)|a_n(t)> for harmonic
    oscillator wave functions a_n(t) and a_m(t), because these functions are real valued.

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
    imuh = np.zeros((n,n), dtype=np.float64)
    for idet in range(0, nd): # run over detector pixels -> (qy, qx)
        jdet = dethash[idet] # (qy, qx) indices in the original grid
        jq2 = jdet[0] + iqey # index of qy in the extended array
        jq1 = jdet[1] + iqex # index of qx in the extended array
        pq = tmx[jq1] * tmy[jq2] # tmx(qx) * tmy(qy)
        feqq = feq[jq2, jq1] # fe(q)
        #
        # off-setting the h grid by -q of the detector pixel is not trivial
        # * given a detector pixel jdet, these indices are frequency + (n>>1)
        # * given an h-grid pixel i, these indices are also the frequency + (n>>1)
        # * now calculate the target frequency h - q, this is simply i - jdet,
        #   however, this is now a frequency and not an index, so we need to add
        #   (n>>1) to get to the target index in the h-grid
        #   j = i - jdet + (n>>1)
        # * apply the offset iqe to get to the index in the extended grid
        #   jj = j + iqe = i - jdet + (n>>1) + iqe
        # * this means, the offset of h - q in the extended grid is
        #   iqe + (n>>1) - jdet
        # 
        #jqh20 = jq2 - n2 # offset of the hy - qy grid in the extended grid
        #jqh10 = jq1 - n2 # offset of the hx - qx grid in the extended grid
        jqh20 = iqey + n2 - jdet[0] # offset of the hy - qy grid in the extended grid
        jqh10 = iqex + n2 - jdet[1] # offset of the hy - qy grid in the extended grid
        #
        imuh[:,:] = 0.0 # reset loop result
        # run through the h vectors and sum to output
        # Disabling pylint warning, see https://github.com/PyCQA/pylint/issues/2910
        for i2 in prange(0, n): # pylint: disable=not-an-iterable # hy (parallel)
            j2 = i2 + jqh20 # hy --> hy - qy
            for i1 in range(0, n): # hx
                j1 = i1 + jqh10 # hx --> hx - qx
                pqh = tmy[j2] * tmx[j1] # matrix elements for h - q
                p = np.double((pq * pqh).real) # product of 2d matrix elements -> is always real valued but positive or negative
                imuh[i2,i1] = p * feq[j2,j1] * feqq # store on h-grid per detector pixel
        #
        muh[0:n,0:n] += imuh[0:n,0:n] # avoid racing condition by accumulating out of the parallel loop
        #
    return 0


def npint_muh(l_qi, iqex, iqey, det, tmx, tmy, feq):
    '''
    npint_muh

    Calculates mu_h for a given pair of matrix elements tmx, tmy, detector
    function det and electron scattering factors feq.
    This is for a local approximation in which det(q) is 1 for all q.
    The approximation becomes bad for smaller detectors, i.e. when only
    a few det(q) = 1 and most others are 0.

    mu_{n,m}(h) = sum_q det(q) fe(q-h) fe(q) <a_n(t)|exp(2pi I (q-h).t)|a_m(t)> <a_n(t)|exp(-2pi I q.t)|a_m(t)>

    This is vor vectors h, q, t, m, and n and the sum over q is performed on shifted
    grids q-h and q.
    Vector m is the initial state and n is the final state set of oscillator quantum numbers.
    Transition matrix elements tmx and tmy are for the two oscillator modes and the 2d
    version is spanned by an outer product here on the extended grid.    

    Note also that
    tm(q) = <a_n(t)|exp(-2pi I q.t)|a_m(t)>
    tm*(q-h) = <a_n(t)|exp(2pi I (q-h).t)|a_m(t)>
    for harmonic oscillator wave functions a_n(t) and a_m(t), because these functions are real valued.

    '''
    tm = np.outer(tmx, tmy) # transition matrix element on the extended grid
    


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
        8) a = pmu.get_mul2d([0.001, 0.01]) # ... to calculate mu(dE,gy,gx)
           a = get_spec_prb(probe, [0.001, 0.01]) # ... to calculate probe EELS

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

    def set_probability_threshold(self, prob_threshold=0.05):
        '''

        Sets a new probability threshold.

        This threshold is applied in calculations to determine relevant
        contributions when averaging over the phonon density of states
        and for the thermal averaging. The threshold is taken relative
        to the largest value of the PDOS or the Boltzmann distribution.

        Parameters
        ----------
            prob_threshold : float, default: 0.05
                relative probability threshold

        Returns
        -------
            None

        '''
        self.pthr = np.abs(prob_threshold) # only positive values

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
        nq2 = nq >> 1 # nyquist number
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
        self.det["grid_fft"] = np.roll(l_det, shift=(-nq2, -nq2), axis=(0, 1))
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
        l_q_rng = [np.amin(l_qi), np.amax(l_qi)] # original grid range of frequencies
        l_det_idr = self.det["hash"]["ifreq_range"] # frequency range of the detector on the q-grid
        l_ext_rng = np.array([l_q_rng, l_q_rng]) # preset extended range by primary range
        # take into account that we extend for frequencies h - q !!!
        # -> extended minimum = minimum of original minimum minus detector maximum
        # -> extended maximum = maximum of original maximum minus detector minimum
        l_ext_rng[0,0] = min(l_ext_rng[0,0], l_ext_rng[0,0] - l_det_idr[0,1]) # update minimum ifreq of y
        l_ext_rng[0,1] = max(l_ext_rng[0,1], l_ext_rng[0,1] - l_det_idr[0,0]) # update maximum ifreq of y
        l_ext_rng[1,0] = min(l_ext_rng[1,0], l_ext_rng[1,0] - l_det_idr[1,1]) # update minimum ifreq of x
        l_ext_rng[1,1] = max(l_ext_rng[1,1], l_ext_rng[1,1] - l_det_idr[1,0]) # update maximum ifreq of x
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

    def prepare_qgrid_tpc(self):
        '''

        prepare_qgrid_ext

        Prepares a q-grid for the transition potential calculation.

        Requires prior calls to members
        set_qgrid

        The size of the grid depends on the original q grid.
        
        This is for square grids only.

        This function is called from class functions and
        should not be used elsewhere. It relies on a full
        setup of the object (grid and detector)

        '''
        n = self.qgrid["n"]
        n2 = n >> 1 # nyquist index
        l_qi = ((np.arange(0, n) + n2) % n) - n2 # list of q frequency indices
        # store tpc grid data
        self.qgrid["tpc"] = {
            "n" : n, # number of pixels in one dimension (square size)
            "dq" : self.qgrid["dq"], # q step size
            "l_qi" : l_qi # y frequency indices
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

    def prepare_feq_tpc(self):
        '''

        prepare_feq_tpc

        Prepares atom scattering factors on the q-grid
        for transition potential calculations.

        This function is called from class functions and
        should not be used elsewhere. It relies on a full
        setup of the object (grid and detector)

        Current implementation is for square grids.

        '''
        gtp = self.qgrid["tpc"]
        n = gtp["n"]
        dq = self.qgrid["dq"]
        l_q2ex = np.sum(np.meshgrid((gtp["l_qi"] * dq)**2, (gtp["l_qi"] * dq)**2), axis=0)
        l_qabs = np.sqrt(l_q2ex)
        gtp["l_feq"] = get_fen(l_qabs.flatten(), self.atom["data"]["q"], self.atom["data"]["fe"]).reshape(n,n).astype(np.double)

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
        pfac = 1.0 # * dq * dq # * (ec.PHYS_HPL**2 / (2*np.pi * ec.EL_M0))**2 # constant prefactor (h^2 / (2 pi m_el))^2 dqx dqy
        # ^^    need to clarify the units here
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
        j = numint_muh(l_qi, l_gsh[1], l_gsh[0], dethash, hx, hy, feq, sdet)
        return sdet * pfac
    
    def get_tpq(self, ep, mx, nx, my, ny):
        '''

        get_tpq

        Calculates the transition potential as a function of q
        for the given phonon energy ep and the transition quantum
        numbers mx -> nx, my -> ny.

        The returned data is missing a prefactor
        h^2 / (2 pi m0)
        where m0 is the electron rest mass.

        Returns an array of values for a square fft based q-grid.
        The output is an array of type numpy.complex128

        '''
        gtp = self.qgrid["tpc"]
        dq = gtp["dq"]
        l_qi = gtp["l_qi"]
        l_q = l_qi * dq # actual spatial frequencies in 1/A
        # get oscillator parameters
        w = ep / ec.PHYS_HBAREV # frequency in Hz
        mat = self.atom["mass"] * ec.PHYS_MASSU # atom mass in kg
        u02 = ho.usqr0(mat, w) * 1.E20 # ground state MSD in A^2
        # calculate transition potential -> h
        hx = ho.tsq(l_q, u02, mx, nx) # x mode transition factors <a_nx|H|a_mx>(q)
        hy = ho.tsq(l_q, u02, my, ny) # y mode transition factors <a_ny|H|a_my>(q)
        h = np.outer(hy, hx) * gtp["l_feq"] # H(qy,qx) = Hx(qx) * Hy(qy) * fe(qy,qx)
        return h

    def get_trstr_tp(self, probe, ep, mx, nx, my, ny):
        '''

        get_trstr_tp

        Calculates the transition strength for a given probe
        wavefunction and the current detector.

        This function is called from class functions and
        should not be used elsewhere. It relies on a full
        setup of the object (grid and detector)

        '''
        l_det = self.det["grid_fft"]
        pfac = 1.0 # * dq * dq # * (ec.PHYS_HPL**2 / (2*np.pi * ec.EL_M0))**2 # constant prefactor (h^2 / (2 pi m_el))^2 dqx dqy
        # ^^    need to clarify the units here
        # calculate transition potential -> h
        h = self.get_tpq(ep, mx, nx, my, ny)
        # calculate inelastic wave function (multiply h in real space)
        pinel = probe * np.fft.ifft2(h) # ... (missing I sigma ) ...
        pinelq = np.fft.fft2(pinel) # inelastic wavefunction in q-space
        # inelastic diffraction pattern
        difpat = pinelq.real**2 + pinelq.imag**2
        # ... integration over the detector area
        #sdet = np.dot(difpat.flatten(), l_det.flatten())
        sdet = np.sum(difpat*l_det)
        return sdet * pfac
    
    def get_trstr_tp_ms(self, probe, ep, mx, nx, my, ny):
        '''

        get_trstr_tp_ms

        Calculates the transition strength for a given probe
        wavefunction and the current detector for multiple
        inelastic scattering.

        This function is called from class functions and
        should not be used elsewhere. It relies on a full
        setup of the object (grid and detector)

        '''
        l_det = self.det["grid_fft"]
        nms = len(ep) # number of scattering events (assume all other input is likewise of that length)
        assert nms > 0, 'This requires list inputs of at least length 1.'
        pfac = 1.0 # * dq * dq # * (ec.PHYS_HPL**2 / (2*np.pi * ec.EL_M0))**2 # constant prefactor (h^2 / (2 pi m_el))^2 dqx dqy
        # ^^    need to clarify the units here
        ndim = np.array(probe.shape)
        pfac_pix = np.sqrt(ndim[0] * ndim[1])
        # calculate transition potential -> h
        h = self.get_tpq(ep[0], mx[0], nx[0], my[0], ny[0])
        # calculate inelastic wave function after first transition (real-space)
        pinel = probe * np.fft.ifft2(h) # ... (missing I sigma ) ...
        if (nms > 1):
            for ims in range(1, nms):
                # calculate transition potential -> h
                h = self.get_tpq(ep[ims], mx[ims], nx[ims], my[ims], ny[ims])
                pinel = (pinel * np.fft.ifft2(h) * pfac_pix) # ... apply next TP in real space
        # calculate inelastic wave function (multiply h in real space)
        pinelq = np.fft.fft2(pinel) # inelastic wavefunction in q-space
        # inelastic diffraction pattern
        difpat = pinelq.real**2 + pinelq.imag**2
        # ... integration over the detector area
        #sdet = np.dot(difpat.flatten(), l_det.flatten())
        sdet = np.sum(difpat*l_det)
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
        are the q-grid (qy, qx) defined by set_qgrid.

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
        a_mu = np.zeros((len(l_dE), self.qgrid["n"], self.qgrid["n"]), dtype=np.double) # output array
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
        ntr_total = 0 # count number of all transitions
        for iep in range(0, len(self.pdos["data"]["energy"])): # loop over phonon energies
            pdos = self.pdos["data"]["pdos"][iep] # pdos value in the current phonon energy bin
            if pdos < pdos_thr: # check pdos against probability threshold
                continue # skip phonon energy
            ep = self.pdos["data"]["energy"][iep] # current phonon energy
            if verbose > 0: print('(get_mul2d): calculating contributions for phonon energy {:.4f} eV (pdos = {:.2f}%) ...'.format(ep, pdos*100.))
            w = ep / ec.PHYS_HBAREV # oscillator frequency
            nimax = nmaxt(ep, self.t, self.pthr) # get max. initial state quantum number to take into account
            if verbose > 0: print('(get_mul2d): including initial states up to m = {:d} ...'.format(nimax))
            nrmtbd = (np.exp(ep/self.tev) - 1)**2 / np.exp(ep/self.tev) # normalization factor for the 2-d boltzmann distribution
            # boltzmann distribution threshold
            pb_thr = nrmtbd * pbolz(ep, self.t) * self.pthr # 2d ground state occupation time relative threshold
            if verbose > 0: print('(get_mul2d): allowing 2d Boltzmann factors above {:.2f}% ...'.format(pb_thr*100.))
            # per phonon energy (to be weighted by pdos)
            for niy in range(0, nimax+1): # loop over initial states in the y dimension
                eiy = ho.En(niy, w) / ec.PHYS_QEL # initial y state energy in eV
                pby = pbolz(eiy, self.t) # initial y state Boltzmann distribution probability
                for nix in range(0, nimax+1): # loop over initial states in the x dimension
                    eix = ho.En(nix, w) / ec.PHYS_QEL # initial x state energy in eV
                    pbx = pbolz(eix, self.t) # initial x state Boltzmann distribution probability
                    pb2 = nrmtbd * pbx * pby # normalized occupation of the initial state
                    if pb2 < pb_thr: continue # 
                    if verbose > 1: print('(get_mul2d): * [{:.4f} eV] initial state [{:d},{:d}] (pbol = {:.2f}%) ...'.format(ep,nix,niy,pb2*100.))
                    #
                    # Initialize an expanding loop over final states.
                    # This loop will at least include single phonon excitations.
                    # It will terminate expanding x and y final states independently.
                    # The termination is made when during the current step of expansion
                    # only transitions smaller than a threshold self.pthr compared
                    # to the previous maximum was encountered.
                    # We increase the LEVEL OF EXCITATION dn = n - m.
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
                        ntr0 = ntr # transitions before this round
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
                        if not single_phonon: # corners are always multi-phonon excitations
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
                            if verbose > 2: print('(get_mul2d):   * y transition level raised to {:d}'.format(nyd))
                            trsmax = max(trsmax, trsmaxy) # maximum update
                        if bmorefx:
                            nxd += 1 # add columnss
                            if verbose > 2: print('(get_mul2d):   * x transition level raised to {:d}'.format(nxd))
                            trsmax = max(trsmax, trsmaxx) # maximum update
                        #
                        if ntr == ntr0: # stop, because no transition was added, we do not expect more to come
                            # this catches infinite loops because there are skipping events in the above
                            bmorefy = False
                            bmorefx = False
                        #
                        if single_phonon: # single-phonon transition case, just stop here
                            bmorefy = False
                            bmorefx = False
                        #
                    #
                    ntr_total += ntr # sum total transitions
                    if verbose > 1: print('(get_mul2d): * transitions added: {:d}'.format(ntr))
                    #
                #
            #
        #
        if verbose > 0: print('(get_mul2d): total number of transitions considered: {:d}'.format(ntr_total))
        # returning
        return {
            "data" : a_mu,
            "l_dE" : l_dE,
            "l_q" : self.qgrid["l_q"]
        }

    def get_spec_prb(self, probe, energy_loss_range, single_phonon=False, double_scattering_level=0, verbose=0):
        '''

        get_spec_prb

        Calculates a phonon EEL spectrum for a given probe.

        The calculation is performed for the current atom, detector,
        PDOS, temperature, and q-grid in a local approximation.
        
        The output will be a 1-dimensional array of intensities,
        sampling the requested energy-loss range in multiple steps
        of the internal PDOS sampling set with the method set_pdos.

        The input probe wave function is assumed to be in real space
        and normalized on a grid reciprocal to the q-grid (qy, qx)
        defined by set_qgrid.

        Parameters
        ----------
            probe : 2d numpy array, dtype=np.complex128
                probe wavefunction in real space
            energy_loss_range : array-like, len=2, type float
                lower and upper bound of energy-losses in eV
            single_phonon : boolean, default: False
                flag: limits to single-phonon excitations
            double_scattering_level : int, default: 0
                limits the phonon excitation levels of double-scattering
            verbose : int, default: 0
                verbosity level for text output
            
        Returns
        -------
            dict
                "data" : numpy.ndarray, num_dim=1, dtype=np.float64
                    phonon EELS spectrum single scattering only
                "data_sphon" : numpy.ndarray, num_dim=1, dtype=np.float64
                    phonon EELS spectrum single-phonon excitations only
                "data_dmuls" : numpy.ndarray, num_dim=1, dtype=np.float64
                    phonon EELS spectrum due to double scattering but
                    single-phonon excitations only
                "l_dE" : numpy.ndarray, num_dim=1, dtype=float
                    energy-loss grid in eV

        '''
        assert energy_loss_range[0] < energy_loss_range[1], "Invalid energy loss range"
        # setup the energy loss grid
        dE = self.pdos["data"]["energy_step"] # set energy step from set_pdos
        pdos_thr = self.pthr * np.amax(self.pdos["data"]["pdos"])
        imin = int(energy_loss_range[0] / dE)
        imax = int(energy_loss_range[1] / dE)
        l_dE = np.arange(imin, imax+1, 1) * dE # energy loss grid
        # setup the output arrays
        a_eels = np.zeros(len(l_dE), dtype=np.float64) # total
        a_eels_sphon = a_eels * 0.0 # single phonon scattering & single inelastic scattering
        a_eels_dmuls = a_eels * 0.0 # single inelastic scattering
        # prepare the calculation
        self.prepare_qgrid_tpc() # q-grid used in tp calculations is for fft
        if verbose > 0: print('(get_spec_prb): calculating scattering factors ...')
        self.prepare_feq_tpc() # prepares fe(q) on the fft q-grid

        # 2) Oscillators and states
        ntr_total = 0 # count number of all transitions
        #
        # singe inelastic scattering (processes with one transition potential)
        # but including multi-phonon excitations (when single_phonon==False)
        for iep in range(0, len(self.pdos["data"]["energy"])): # loop over phonon energies
            pdos = self.pdos["data"]["pdos"][iep] # pdos value in the current phonon energy bin
            if pdos < pdos_thr: # check pdos against probability threshold
                continue # skip phonon energy
            ep = self.pdos["data"]["energy"][iep] # current phonon energy
            if verbose > 0: print('(get_spec_prb): calculating contributions for phonon energy {:.4f} eV (pdos = {:.2f}%) ...'.format(ep, pdos*100.))
            w = ep / ec.PHYS_HBAREV # oscillator frequency
            nimax = nmaxt(ep, self.t, self.pthr) # get max. initial state quantum number to take into account
            if verbose > 0: print('(get_spec_prb): including initial states up to m = {:d} ...'.format(nimax))
            nrmtbd = (np.exp(ep/self.tev) - 1)**2 / np.exp(ep/self.tev) # normalization factor for the 2-d boltzmann distribution
            # boltzmann distribution threshold
            pb_thr = nrmtbd * pbolz(ep, self.t) * self.pthr # 2d ground state occupation time relative threshold
            if verbose > 0: print('(get_spec_prb): allowing 2d Boltzmann factors above {:.2f}% ...'.format(pb_thr*100.))
            # per phonon energy (to be weighted by pdos)
            for niy in range(0, nimax+1): # loop over initial states in the y dimension
                eiy = ho.En(niy, w) / ec.PHYS_QEL # initial y state energy in eV
                pby = pbolz(eiy, self.t) # initial y state Boltzmann distribution probability
                for nix in range(0, nimax+1): # loop over initial states in the x dimension
                    eix = ho.En(nix, w) / ec.PHYS_QEL # initial x state energy in eV
                    pbx = pbolz(eix, self.t) # initial x state Boltzmann distribution probability
                    pb2 = nrmtbd * pbx * pby # normalized occupation of the initial state
                    if pb2 < pb_thr: continue # skip due to low initial state occupation probability
                    if verbose > 1: print('(get_spec_prb): * [{:.4f} eV] initial state [{:d},{:d}] (pbol = {:.2f}%) ...'.format(ep,nix,niy,pb2*100.))
                    #
                    # Initialize an expanding loop over final states.
                    # This loop will at least include single phonon excitations.
                    # It will terminate expanding x and y final states independently.
                    # The termination is made when during the current step of expansion
                    # only transitions smaller than a threshold self.pthr compared
                    # to the previous maximum was encountered.
                    # We increase the LEVEL OF EXCITATION dn = n - m.
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
                        ntr0 = ntr # transitions before this round
                        if bmorefy: # add next rows (excluding corners)
                            trsmaxy = 0.
                            for nfy in [niy-nyd,niy+nyd]: # nfy row indices
                                if nfy < 0: continue # skip negative qn
                                efy = ho.En(nfy, w) / ec.PHYS_QEL # final y state energy
                                for nfx in range(nix-nxd+1,nix+nxd): # loop nfx from corner+1 to corner-1
                                    if nfx < 0: continue # skip negative qn
                                    if (niy == nfy) and (nix == nfx): continue # skip elastic (should actually not happen)
                                    is_sphon = (abs(nfx-nix) + abs(nfy-niy) == 1) # current transition is  single-phonon
                                    if (single_phonon and (not is_sphon)): continue # skip multi-phonons in case of single-phonon calculation
                                    efx = ho.En(nfx, w) / ec.PHYS_QEL # final x state energy
                                    delE = efx - eix + efy - eiy # energy loss of the probing electron
                                    idelE = int(np.rint(delE / dE)) #
                                    if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                                        s = self.get_trstr_tp(probe, ep, nix, nfx, niy, nfy)
                                        trsmaxy = max(trsmaxy, s)
                                        jdelE = idelE - imin
                                        sw = (pdos * pb2 * s)
                                        a_eels[jdelE] += sw
                                        if is_sphon: a_eels_sphon[jdelE] += sw
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
                                    is_sphon = (abs(nfx-nix) + abs(nfy-niy) == 1) # current transition is  single-phonon
                                    if (single_phonon and (not is_sphon)): continue # skip multi-phonons in case of single-phonon calculation
                                    efy = ho.En(nfy, w) / ec.PHYS_QEL # final x state energy
                                    delE = efx - eix + efy - eiy # energy loss of the probing electron
                                    idelE = int(np.rint(delE / dE)) #
                                    if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                                        s = self.get_trstr_tp(probe, ep, nix, nfx, niy, nfy)
                                        trsmaxx = max(trsmaxx, s)
                                        jdelE = idelE - imin
                                        sw = (pdos * pb2 * s)
                                        a_eels[jdelE] += sw
                                        if is_sphon: a_eels_sphon[jdelE] += sw
                                        ntr += 1
                                    #
                            # exit criterion for final states x range
                            if (trsmax > 0.0) and (trsmaxx > 0.0): # global maximum present
                                if trsmaxx < 0.5 * self.pthr * trsmax: # max. on row loop is less than a threshold -> converged cols
                                    bmorefx = False
                        #
                        # handle corners
                        if not single_phonon: # corners are always multi-phonon excitations
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
                                        s = self.get_trstr_tp(probe, ep, nix, nfx, niy, nfy)
                                        trsmax = max(trsmax, s)
                                        jdelE = idelE - imin
                                        a_eels[jdelE] += (pdos * pb2 * s)
                                        ntr += 1
                                    #
                        #
                        # update controls
                        if bmorefy:
                            nyd += 1 # add rows
                            if verbose > 2: print('(get_spec_prb):   * y transition level raised to {:d}'.format(nyd))
                            trsmax = max(trsmax, trsmaxy) # maximum update
                        if bmorefx:
                            nxd += 1 # add columnss
                            if verbose > 2: print('(get_spec_prb):   * x transition level raised to {:d}'.format(nxd))
                            trsmax = max(trsmax, trsmaxx) # maximum update
                        #
                        if ntr == ntr0: # stop, because no transition was added, we do not expect more to come
                            # this catches infinite loops because there are skipping events in the above
                            bmorefy = False
                            bmorefx = False
                        #
                        if single_phonon: # single-phonon transition case, just stop here
                            bmorefy = False
                            bmorefx = False
                        #
                    #
                    ntr_total += ntr # sum total transitions
                    if verbose > 1:
                        print('(get_spec_prb): * transitions added: {:d}'.format(ntr))
                        print('(get_spec_prb): * total (single scattering, single phonon): ({:.4E}, {:.4E})'.format(np.sum(a_eels), np.sum(a_eels_sphon)))
                    #
                #
            #
        #
        #
        if (double_scattering_level > 0) and (len(self.pdos["data"]["energy"]) > 1): # add some multiple inelastic scattering
            #
            # double inelastic scattering (processes with two transition potentials)
            # but only single-phonon excitations
            # * set double scattering pdos threshold
            pdos2_thr = self.pthr * np.amax(self.pdos["data"]["pdos"])**2
            # * shortcut for the excitation level
            dsl = double_scattering_level
            #
            # double inelastic scattering with the same phonon energy
            # (diagonal terms with special sequence of excitation)
            for iep in range(0, len(self.pdos["data"]["energy"]) - 1): # loop over phonon energies
                pdos = self.pdos["data"]["pdos"][iep]**2 # pdos value in the current phonon energy bin squared (two processes)
                ep = self.pdos["data"]["energy"][iep] # current phonon energy
                w = ep / ec.PHYS_HBAREV # oscillator frequency
                nimax = nmaxt(ep, self.t, self.pthr) # get max. initial state quantum number to take into account
                nrmtbd = (np.exp(ep/self.tev) - 1)**2 / np.exp(ep/self.tev) # normalization factor for the 2-d boltzmann distribution
                if pdos < pdos2_thr: # check pdos against combined probability threshold
                    continue # skip phonon energy
                if verbose > 0: print('(get_spec_prb): calculating double excitations for phonon energies {:.4f} and {:.4f} eV (pdos = {:.2f}%) ...'.format(ep, ep, pdos*100.))
                if verbose > 0: print('(get_spec_prb): including initial states up to m = {:d} ...'.format(nimax))
                # boltzmann distribution threshold
                pb_thr = nrmtbd * pbolz(ep, self.t) * self.pthr # 2d ground state occupation time relative threshold
                if verbose > 0: print('(get_spec_prb): allowing 2d Boltzmann factors above {:.2f}% ...'.format(pb_thr*100.))
                # per phonon energy (to be weighted by pdos)
                # first transition
                for niy1 in range(0, nimax+1): # loop over initial states in the y dimension
                    eiy1 = ho.En(niy1, w) / ec.PHYS_QEL # initial y state energy in eV
                    pby1 = pbolz(eiy1, self.t) # initial y state Boltzmann distribution probability
                    for nix1 in range(0, nimax+1): # loop over initial states in the x dimension
                        eix1 = ho.En(nix1, w) / ec.PHYS_QEL # initial x state energy in eV
                        pbx1 = pbolz(eix1, self.t) # initial x state Boltzmann distribution probability
                        pb2 = nrmtbd * pbx1 * pby1 # occupation of the initial state of the first transition
                        if pb2 < pb_thr: continue # skip due to low combined initial state occupation probability
                        if verbose > 1: print('(get_spec_prb): * [{:.4f} eV] initial state of 1st excitation [{:d},{:d}] (pbol = {:.2f}%) ...'.format(ep,nix1,niy1,pb2*100.))
                        # initial state and energy of the first transition
                        ei1 = eix1 + eiy1
                        #
                        ntr = 0
                        for nfy1 in range(niy1-dsl, niy1+dsl+1):
                            if nfy1 < 0: continue
                            niy2 = nfy1
                            for nfx1 in range(nix1-dsl, nix1+dsl+1):
                                if nfx1 < 0: continue
                                nix2 = nfx1
                                if (nfy1==niy1) and (nfx1==niy1): continue # skip no change of state
                                for nfy2 in range(niy2-dsl, niy2+dsl+1):
                                    if nfy2 < 0: continue
                                    efy2 = ho.En(nfy2, w) / ec.PHYS_QEL # final y state 2 energy in eV
                                    for nfx2 in range(nix2-dsl, nix2+dsl+1):
                                        if nfx2 < 0: continue
                                        if (nfy2==niy2) and (nfx2==niy2): continue # skip no change of state
                                        efx2 = ho.En(nfx2, w) / ec.PHYS_QEL # final x state 2 energy in eV
                                        ef2 = efy2 + efx2
                                        delE = ef2 - ei1 # energy loss of the probing electron
                                        idelE = int(np.rint(delE / dE)) #
                                        if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                                            s = self.get_trstr_tp_ms(probe, [ep,ep],
                                                                        [nix1,nix2], [nfx1,nfx2],
                                                                        [niy1,niy2], [nfy1,nfy2])
                                            jdelE = idelE - imin
                                            a_eels_dmuls[jdelE] += (pdos * pb2 * s) # weighted by pdos and pbolz
                                            ntr += 1

                        # # start with final states of the first single-phonon excitation
                        # l_nf1 = [[nix1 + 1, niy1], [nix1, niy1 + 1]] # 2 single phonon excitations
                        # l_ef1 = [ei1 + ep, ei1 + ep] # ... 2 energies
                        # # add possible final states of the first single-phonon de-excitation
                        # if nix1 > 0:
                        #     l_nf1.append([nix1 - 1, niy1])
                        #     l_ef1.append(ei1 - ep)
                        # if niy1 > 0:
                        #     l_nf1.append([nix1, niy1 - 1])
                        #     l_ef1.append(ei1 - ep)
                        # # loop over final states of the first transition, which are the initial states of
                        # # the second transition and then determine the states and energies of the final
                        # # states of the second transition
                        # l_nf2 = []
                        # l_ef2 = []
                        # for idf1 in range(0, len(l_nf1)):
                        #     # second single-phonon excitations
                        #     l_nf2.append([l_nf1[idf1][0] + 1, l_nf1[idf1][1]])
                        #     l_ef2.append(l_ef1[idf1] + ep)
                        #     l_nf2.append([l_nf1[idf1][0], l_nf1[idf1][1] + 1])
                        #     l_ef2.append(l_ef1[idf1] + ep)
                        #     # possible second single-phonon de-excitations
                        #     if l_nf1[idf1][0] > 0:
                        #         l_nf2.append([l_nf1[idf1][0] - 1, l_nf1[idf1][1]])
                        #         l_ef2.append(l_ef1[idf1] - ep)
                        #     if l_nf1[idf1][1] > 0:
                        #         l_nf2.append([l_nf1[idf1][0], l_nf1[idf1][1] - 1])
                        #         l_ef2.append(l_ef1[idf1] - ep)
                        # #
                        # ntr = 0
                        # #
                        # # accumulate transition strengths to the spectrum
                        # for idf1 in range(0, len(l_nf1)):
                        #     nfx1 = l_nf1[idf1][0]
                        #     nfy1 = l_nf1[idf1][1]
                        #     nix2 = nfx1
                        #     niy2 = nfy1
                        #     for idf2 in range(0, len(l_nf2)):
                        #         nfx2 = l_nf2[idf2][0]
                        #         nfy2 = l_nf2[idf2][1]
                        #         ef2 = l_ef2[idf2]
                        #         delE = ef2 - ei1 # energy loss of the probing electron
                        #         idelE = int(np.rint(delE / dE)) #
                        #         if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                        #             s = self.get_trstr_tp_ms(probe, [ep,ep],
                        #                                         [nix1,nix2], [nfx1,nfx2],
                        #                                         [niy1,niy2], [nfy1,nfy2])
                        #             jdelE = idelE - imin
                        #             a_eels_dmuls[jdelE] += (pdos * pb2 * s) # weighted by pdos and pbolz
                        #             ntr += 1
                        # #
                        ntr_total += ntr # sum total transitions
                        if verbose > 1:
                            print('(get_spec_prb): * transitions added: {:d}'.format(ntr))
                            print('(get_spec_prb): * total (double scattering): {:.4E}'.format(np.sum(a_eels_dmuls)))
            #
            #
            # double inelastic scattering with different phonon energies
            # (off-diagonal terms with different weighting and arbitrary sequence of excitation)
            for iep1 in range(0, len(self.pdos["data"]["energy"]) - 1): # loop over 1st phonon energies
                pdos1 = self.pdos["data"]["pdos"][iep1] # pdos value in the current phonon energy bin
                ep1 = self.pdos["data"]["energy"][iep1] # current phonon energy
                w1 = ep1 / ec.PHYS_HBAREV # oscillator frequency
                nimax1 = nmaxt(ep1, self.t, self.pthr) # get max. initial state quantum number to take into account
                nrmtbd1 = (np.exp(ep1/self.tev) - 1)**2 / np.exp(ep1/self.tev) # normalization factor for the 2-d boltzmann distribution
                for iep2 in range(iep1 + 1, len(self.pdos["data"]["energy"])): # loop over 2nd phonon energies, off-diagonal only
                    pdos2 = self.pdos["data"]["pdos"][iep2] # pdos value in the current phonon energy bin
                    ep2 = self.pdos["data"]["energy"][iep2] # current phonon energy
                    w2 = ep2 / ec.PHYS_HBAREV # oscillator frequency
                    nimax2 = nmaxt(ep2, self.t, self.pthr) # get max. initial state quantum number to take into account
                    nrmtbd2 = (np.exp(ep2/self.tev) - 1)**2 / np.exp(ep2/self.tev) # normalization factor for the 2-d boltzmann distribution
                    pdos = pdos1 * pdos2
                    if pdos < pdos2_thr: # check pdos against combined probability threshold
                        continue # skip phonon energy
                    if verbose > 0: print('(get_spec_prb): calculating double excitations for phonon energies {:.4f} and {:.4f} eV (pdos = {:.2f}%) ...'.format(ep1, ep2, pdos1*pdos2*100.))
                    if verbose > 0: print('(get_spec_prb): including initial states up to m1 = {:d} and m2 = {:d} ...'.format(nimax1,nimax2))
                    nrmtbd = nrmtbd1 * nrmtbd2  # combined normalization factor for the 2-d boltzmann distributions
                    # boltzmann distribution threshold
                    pb2_thr = nrmtbd * pbolz(ep1, self.t) * pbolz(ep2, self.t) * self.pthr # 2d ground state occupation time relative threshold
                    if verbose > 0: print('(get_spec_prb): allowing 4d Boltzmann factors above {:.2f}% ...'.format(pb2_thr*100.))
                    # per phonon energy (to be weighted by pdos)
                    for niy1 in range(0, nimax1+1): # loop over initial states in the y dimension
                        eiy1 = ho.En(niy1, w1) / ec.PHYS_QEL # initial y state energy in eV
                        pby1 = pbolz(eiy1, self.t) # initial y state Boltzmann distribution probability
                        for nix1 in range(0, nimax1+1): # loop over initial states in the x dimension
                            eix1 = ho.En(nix1, w1) / ec.PHYS_QEL # initial x state energy in eV
                            pbx1 = pbolz(eix1, self.t) # initial x state Boltzmann distribution probability
                            pb21 = pbx1 * pby1 # occupation of the initial state
                            for niy2 in range(0, nimax2+1): # loop over initial states in the y dimension
                                eiy2 = ho.En(niy2, w2) / ec.PHYS_QEL # initial y state energy in eV
                                pby2 = pbolz(eiy2, self.t) # initial y state Boltzmann distribution probability
                                for nix2 in range(0, nimax2+1): # loop over initial states in the x dimension
                                    eix2 = ho.En(nix2, w1) / ec.PHYS_QEL # initial x state energy in eV
                                    pbx2 = pbolz(eix2, self.t) # initial x state Boltzmann distribution probability
                                    pb22 = pbx2 * pby2 # occupation of the initial state
                                    pb4 = nrmtbd * pb21 * pb22 # normalized combined occupation
                                    if pb4 < pb2_thr: continue # skip due to low combined initial state occupation probability
                                    if verbose > 1: print('(get_spec_prb): * [({:.4f},{:.4f}) eV] initial state ([{:d},{:d}],[{:d},{:d}]) (pbol = {:.2f}%) ...'.format(ep1,ep2,nix1,niy1,nix2,niy2,pb4*100.))
                                    #
                                    ei_all = eix1 + eiy1 + eix2 + eiy2 # inital state energy
                                    #
                                    ntr = 0
                                    #
                                    for nfy1 in range(niy1-dsl, niy1+dsl+1):
                                        if nfy1 < 0: continue
                                        efy1 = ho.En(nfy1, w1) / ec.PHYS_QEL # final y state 2 energy in eV
                                        for nfx1 in range(nix1-dsl, nix1+dsl+1):
                                            if nfx1 < 0: continue
                                            if (nfy1==niy1) and (nfx1==niy1): continue # skip no change of state
                                            efx1 = ho.En(nfx1, w1) / ec.PHYS_QEL # final y state 2 energy in eV
                                            for nfy2 in range(niy2-dsl, niy2+dsl+1):
                                                if nfy2 < 0: continue
                                                efy2 = ho.En(nfy2, w2) / ec.PHYS_QEL # final y state 2 energy in eV
                                                for nfx2 in range(nix2-dsl, nix2+dsl+1):
                                                    if nfx2 < 0: continue
                                                    if (nfy2==niy2) and (nfx2==niy2): continue # skip no change of state
                                                    efx2 = ho.En(nfx2, w2) / ec.PHYS_QEL # final x state 2 energy in eV
                                                    ef_all = efy1 + efx1 + efy2 + efx2
                                                    delE = ef_all - ei_all # energy loss of the probing electron
                                                    idelE = int(np.rint(delE / dE)) #
                                                    if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                                                        s = self.get_trstr_tp_ms(probe, [ep1,ep2],
                                                                                    [nix1,nix2], [nfx1,nfx2],
                                                                                    [niy1,niy2], [nfy1,nfy2])
                                                        jdelE = idelE - imin
                                                        a_eels_dmuls[jdelE] += (pdos * pb2 * s * 4.0) # weighted by pdos and pbolz * 4 due to two possibilities
                                                        ntr += 1


                                    # l_schemei = np.array([[nix1, niy1], [nix2, niy2]], dtype=int) # initial state scheme
                                    # l_energyi = np.array([[eix1, eiy1], [eix2, eiy2]], dtype=float) # initial state energies
                                    # l_freq = np.array([w1, w2], dtype=float) # initial state frequencies
                                    # ei_all = np.sum(l_energyi) # total initial state energy
                                    # #
                                    # # final states loop (single_phonon only)
                                    # ntr = 0
                                    # # excitations first
                                    # for ise in range(0, 2): # loop over scattering events
                                    #     for imd in range(0, 2): # loop over modes
                                    #         l_schemef = l_schemei.copy()
                                    #         l_energyf = l_energyi.copy()
                                    #         l_schemef[ise,imd] += 1 # raise final state of current mode
                                    #         l_energyf[ise,imd] = ho.En(l_schemef[ise,imd], l_freq[ise]) / ec.PHYS_QEL # changed final state energy
                                    #         ef_all = np.sum(l_energyf)
                                    #         delE = ef_all - ei_all # energy loss of the probing electron
                                    #         idelE = int(np.rint(delE / dE)) #
                                    #         if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                                    #             s = self.get_trstr_tp_ms(probe, [ep1,ep2],
                                    #                                         [nix1,nix2], [l_schemef[0][0], l_schemef[1][0]],
                                    #                                         [niy1,niy2], [l_schemef[0][1], l_schemef[1][1]])
                                    #             jdelE = idelE - imin
                                    #             a_eels_dmuls[jdelE] += (pdos * pb4 * s * 4.0) # weight 4 due to possible two sequences
                                    #             ntr += 1
                                    # # de-excitations second
                                    # for ise in range(0, 2): # loop over scattering events
                                    #     for imd in range(0, 2): # loop over modes
                                    #         if l_schemei[ise,imd] == 0: continue # skip ground states
                                    #         l_schemef = l_schemei.copy()
                                    #         l_energyf = l_energyi.copy()
                                    #         l_schemef[ise,imd] -= 1 # raise final state of current mode
                                    #         l_energyf[ise,imd] = ho.En(l_schemef[ise,imd], l_freq[ise]) / ec.PHYS_QEL # changed final state energy
                                    #         ef_all = np.sum(l_energyf)
                                    #         delE = ef_all - ei_all # energy loss of the probing electron
                                    #         idelE = int(np.rint(delE / dE)) #
                                    #         if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                                    #             s = self.get_trstr_tp_ms(probe, [ep1,ep2],
                                    #                                         [nix1,nix2], [l_schemef[0][0], l_schemef[1][0]],
                                    #                                         [niy1,niy2], [l_schemef[0][1], l_schemef[1][1]])
                                    #             jdelE = idelE - imin
                                    #             a_eels_dmuls[jdelE] += (pdos * pb4 * s * 4.0) # weight 2 due to possible two sequences
                                    #             ntr += 1
                                    #
                                    ntr_total += ntr # sum total transitions
                                    if verbose > 1:
                                        print('(get_spec_prb): * transitions added: {:d}'.format(ntr))
                                        print('(get_spec_prb): * total (double scattering): {:.4E}'.format(np.sum(a_eels_dmuls)))
        #
        if verbose > 0:
            print('(get_spec_prb): total number of transitions considered: {:d}'.format(ntr_total))
            print('(get_spec_prb): * total (single scattering): {:.4E}'.format(np.sum(a_eels)))
            print('(get_spec_prb): * total (single-phonon): {:.4E}'.format(np.sum(a_eels_sphon)))
            print('(get_spec_prb): * total (double scattering): {:.4E}'.format(np.sum(a_eels_dmuls)))

        # returning
        return {
            "data" : a_eels,
            "data_sphon" : a_eels_sphon,
            "data_dmuls" : a_eels_dmuls,
            "l_dE" : l_dE
        }