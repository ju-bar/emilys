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
import emilys.optics.probe as prb
from emilys.optics.aperture import aperture
from emilys.numerics.rngdist import rngdist
from emilys.structure.atomtype import Z_from_str

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

@jit(float64(float64, float64, float64, float64), nopython=True)
def gauss_peak(x, amp, pos, sig):
    dx2 = (x-pos)**2
    ts2 = 2 * sig**2
    return amp * np.exp(-dx2 / ts2) / np.sqrt( np.pi * ts2)

# define spectrum function with system energy resolution
def get_spec_conv(l_e, l_y, E0, E1, dE, Eres):
    n1 = len(l_e)
    e_sig = Eres / np.sqrt(8. * np.log(2.0))
    l_es = np.arange(E0, E1 + 0.5*dE, dE)
    n2 = len(l_es)
    l_ys = np.zeros(n2, dtype=l_y.dtype)
    ys = gauss_peak(0.,1.,0.,1.) # compile peak function
    for i in range(0, n1):
        delta_e = 0.
        if i > 0: delta_e += (l_e[i] - l_e[i-1])*0.5
        if i < n1-1: delta_e += (l_e[i+1] - l_e[i])*0.5
        for j in range(0, n2):
            ys = gauss_peak(l_es[j], l_y[i]*delta_e, l_e[i], e_sig)
            l_ys[j] += ys
    return l_es, l_ys

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



@jit(int64(int64[:], int64, int64, int64[:,:], float64[:], complex128[:], complex128[:], float64[:,:], float64[:,:]), nopython=True, parallel=True)
def numint_muh(l_qi, iqex, iqey, dethash, detval, tmx, tmy, feq, muh):
    '''
    numint_muh

    Calculates mu_h for a given pair of matrix elements tmx, tmy, detector
    function dethash and electron scattering factors feq into muh.

    mu_{n,m}(h) = sum_q fe(h-q) fe(q) <a_n(t)|exp(-2pi I (h-q).t)|a_m(t)> <a_m(t)|exp(-2pi I q.t)|a_n(t)>

    This is vor vectors h, q, t, m, and n and the sum over q is performed for grid
    points in the detector collection aperture (given by dethash). Vector m is the
    initial state and n is the final state set of oscillator quantum numbers.

    Note: The integration step size (step area) needs to be multiplied still.

    Note also that
    <a_n(t)|exp(-2pi I q.t)|a_m(t)> = <a_m(t)|exp(-2pi I q.t)|a_n(t)> for harmonic
    oscillator wave functions a_n(t) and a_m(t), because these functions are real valued.

    Compiled code using jit.
    int64(int64[:], int64, int64, int64[:,:], float64[:], complex128[:], complex128[:], float64[:,:], float64[:,:])

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
        detval: float64[:], shape(nd)
            detector sensitivity for each pixel of the table dethash
        tmx : complex128[:], shape(nextended_x)
            transition matrix element of the x mode, calculated on the extended grid
        tmy : complex128[:], shape(nextended_y)
            transition matrix element of the y mode, calculated on the extended grid
        feq: float64[:,:], shape(nextended_y,nextended_x)
            electron scattering factor on the extended grid
        muh: float64[:,:]
            resulting mu_{h,0} for h frequencies of the original grid,
            the integration step size (step area) needs to be multiplied still

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
        sdet = detval[idet] # detector sensitivity
        jq2 = jdet[0] + iqey # index of qy in the extended array
        jq1 = jdet[1] + iqex # index of qx in the extended array
        pq = tmx[jq1] * tmy[jq2] # tmx(qx) * tmy(qy)
        feqq = feq[jq2, jq1] # fe(q)
        #
        # off-setting the h grid by -q of the detector pixel is not trivial
        # * given a detector pixel jdet, these indices are frequency + (n>>1)
        # * given an h-grid pixel i, these indices are also the frequency + (n>>1)
        # * now calculate the target frequency h - q, this is simply i - jdet,
        #   however, this is a frequency and not an index, so we need to add
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
        jqh10 = iqex + n2 - jdet[1] # offset of the hx - qx grid in the extended grid
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
        muh[0:n,0:n] += (imuh[0:n,0:n] * sdet) # avoid racing condition by accumulating out of the parallel loop
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
        rngdist : dict
            random number generators for positional configurations

    Members
    -------
        set_qgrid : Set q-grid parameters for numerical calculation (square size)
        set_atom : Set atom data and prepares scattering factors
        set_detector : Set detector data and prepare related functions
        set_temperatur : Set the temperature to assume in calculations
        set_pdos : Set a phonon density of states (PDOS)
        get_utsqr : Returns the mean squared displacement for the current atom and PDOS at a temperature
        get_utsqr_ho : Returns the mean squared displacement for the current atom at a temperature and an Einstein energy
        get_sdf : Returns a spectral distribution function for an atom, the PDOS and a temperature

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
            "Z" : 0,
            "symbol" : "",
            "mass" : 0.0
        }
        self.rngdist = { # random number generators for positional configurations
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
        l_detvals = []
        q0 = self.det["center"]
        rq0 = np.array([q0[1],q0[0]], dtype=np.float64)
        qd0 = self.det["inner"]
        qd1 = self.det["outer"]
        psmt = np.float64(self.det["edge_smooth"] * (l_q[1] - l_q[0]))
        #l_dqy2 = (l_q - q0[0])**2
        #l_dqx2 = (l_q - q0[1])**2
        iqr = [[nq,-nq],[nq,-nq]] # min and max frequencies of the detector in rows and columns
        dthr = self.det["transmission_threshold"]
        for i in range(0, nq): # qy
            qy = l_q[i]
            for j in range(0, nq): # qx
                qx = l_q[j]
                vq = np.array([qx,qy], dtype=np.float64)
                vi = 0.0
                if qd0 > 0:
                    vi = aperture(vq, rq0, np.float64(qd0), psmt)
                va = aperture(vq, rq0, np.float64(qd1), psmt)
                v = va - vi
            
                #q = np.sqrt(l_dqy2[i] + l_dqx2[j])
                #if (q >= qd0) and (q < qd1):

                if v > dthr:
                    l_det[i,j] = v
                    l_dethash.append([i,j])
                    l_detvals.append(v)
                    iqr[0][0] = min(iqr[0][0], l_qi[i])
                    iqr[0][1] = max(iqr[0][1], l_qi[i])
                    iqr[1][0] = min(iqr[1][0], l_qi[j])
                    iqr[1][1] = max(iqr[1][1], l_qi[j])
        self.det["grid"] = l_det
        self.det["grid_fft"] = np.roll(l_det, shift=(-nq2, -nq2), axis=(0, 1))
        self.det["hash"] = {
            "index" : np.array(l_dethash, dtype=np.int64),
            "value" : np.array(l_detvals, dtype=np.float64),
            "ifreq_range" : np.array(iqr, dtype=np.int64)
        }

    def set_detector(self, inner, outer, center, det_smooth=1.0, det_threshold=0.001):
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
            det_smooth : float, default: 0.5
                detector edge smoothness in fractions of the q grid step
            det_threshold : float, default: 0.001
                detector transmisson threshold
        
        Returns
        -------
            None
        
        '''
        self.det["inner"] = inner
        self.det["outer"] = outer
        self.det["center"] = center
        self.det["edge_smooth"] = det_smooth
        self.det["transmission_threshold"] = det_threshold
        self.prepare_detector()

    def set_temperature(self, T):
        '''

        set_temperature

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

    def get_pbol(self, ev):
        nrm = (np.exp(ev/self.tev) - 1) / np.exp(0.5 * ev / self.tev)
        nimax = nmaxt(ev, self.t, self.pthr) # get max. initial state quantum number to take into account
        l_m = np.arange(0, nimax+1, 1, dtype=int) # state quantum number series
        w = ev / ec.PHYS_HBAREV # oscillator frequency in Hz
        l_en = ho.En(l_m, w).astype(np.float64) / ec.PHYS_QEL # state energy in eV
        l_p = nrm * pbolz(l_en, self.t) # Boltzmann distribution probability
        psum1 = np.sum(l_p) # get sum of probabilities
        l_p = l_p / psum1 # renormalize to account for cut-off series
        l_c = l_p * 0.0 # initialize cdf
        l_c[0] = l_p[0]
        for i in range(1, len(l_p)):
            l_c[i] = l_c[i-1] + l_p[i]
        return {  'phonon_energy' : ev, 'nmax' : nimax, 'frequency' : w
                , 'energies' : l_en
                , 'quantum_numbers' : l_m
                , 'pdf' : l_p 
                , 'cdf' : l_c}
    
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
        assert dE > 0.00001, "Expecting non-negative energy step size dE."
        # backup input pdos
        self.pdos["original"] = {
            "energy" : l_ev,
            "pdos" : l_pdos
        }
        # resample the pdos on a fine energy grid
        dEfine = sub_sample * dE
        s1, x1, y1 = resample_pdos(l_ev, l_pdos, dEfine, pdos_ip_kind=pdos_ip_kind)
        self.pdos["original"]["norm"] = s1
        # store probability density on the fine sampling
        self.pdos["resampled"] = {
            "energy_step" : dEfine,
            "ip_kind" : pdos_ip_kind,
            "energy" : x1,
            "pdos" : y1
        }
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
        #
        # store integrated pdos data (these are probabilities, not probability densities)
        self.pdos["data"] = {
            "energy_step" : dE,
            "energy" : x2[1:nep-1],
            "pdos" : y2[1:nep-1],
            "ip_kind" : pdos_ip_kind,
            "norm" : self.pdos["original"]["norm"]
        }

    def get_pdos(self):
        l_p = self.pdos["data"]["pdos"]
        l_e = self.pdos["data"]["energy"]
        emax = l_e[-1] + self.pdos["data"]["energy_step"]
        if l_e[0] > 0.0:
            l_p = np.concatenate((np.array([0.]), l_p))
            l_e = np.concatenate((np.array([0.]), l_e))
        if l_p[-1] > 0.0:
            l_p = np.concatenate((l_p, np.array([0.])))
            l_e = np.concatenate((l_e, np.array([emax])))
        psum = np.sum(l_p)
        l_p = l_p / psum
        l_c = l_p * 0.0 # initialize cdf
        l_c[0] = l_p[0]
        for i in range(1, len(l_p)):
            l_c[i] = l_c[i-1] + l_p[i]
        return {
            "energy_step" : self.pdos["data"]["energy_step"],
            "energy" : l_e,
            "pdf" : l_p,
            "cdf" : l_c,
            "norm" : self.pdos["data"]["norm"]
        }
    
    def get_pdos_pdf(self, dEp):
        pdos = self.pdos["resampled"]
        l_p = pdos["pdos"]
        l_e = pdos["energy"]
        Ep_max = np.amax(l_e)
        pdos_ip1 = interpolate.interp1d(l_e, l_p, kind=pdos['ip_kind'],copy=True,bounds_error=False,fill_value=0.0)
        #
        l_e1 = np.arange(dEp, Ep_max + dEp/2, dEp)
        l_p1 = np.abs(pdos_ip1(l_e1)) # interpolated PDOS values
        p1tot = np.trapz(l_p1, x=l_e1) # total (integrated on the grid)
        l_pn = l_p1 / p1tot # normalized PDOS on the fine grid [meV^{-1]]
        return {
            "energy_step" : dEp,
            "energy" : l_e1,
            "pdf" : l_pn,
            "norm" : self.pdos["data"]["norm"]
        }

    def write_pdos_file(self, sfile, dEp, symbol="", temperature=-1.0):
        if len(sfile) == 0: return 1
        if dEp <= 0.0: return 2
        # atom data
        symb = symbol
        if len(symb) == 0: symb = self.atom["symbol"]
        ma = self.atom["mass"]
        # temperature
        temp = temperature
        if temp < 0.0: temp = self.t
        # PDOS
        pdos = self.get_pdos_pdf(dEp)
        l_p = pdos["pdf"]
        l_e = pdos["energy"]
        n = len(l_e)
        i0 = 0
        i1 = n-1
        for i in range(0, n):
            if l_p[i] > 0.0:
                i0 = i
                break
        for i in range(n-1, -1, -1):
            if l_p[i] > 0.0:
                i1 = i
                break
        #
        with open(sfile, 'w') as f:
            f.write('PDOS\n')
            f.write(symb + '\n')
            f.write('{:.5f}\n'.format(ma))
            f.write('{:.1f}\n'.format(temp))
            f.write('{:d}\n'.format(1 + i1 - i0))
            for i in range(i0, i1+1, 1):
                f.write('{:.6f}   {:.6f}\n'.format(l_e[i], l_p[i]))
            f.write('\n')
            f.close()
        #
        return 0

    def load_pdos_file(self, sfile):
        nl = 0
        lines = []
        with open(sfile, 'r') as f:
            lines = f.readlines()
            f.close()
            nl = len(lines)
        if nl > 0:
            i0 = -1
            for i in range(0, nl):
                if "PDOS" in lines[i]: i0 = i
            if i0 >= 0:
                symb = lines[i0+1]
                Znum = Z_from_str(symb)
                mass = float(lines[i0+2])
                temp = float(lines[i0+3])
                nums = int(lines[i0+4])
                ldos = []
                if nums > 0:
                    i1 = i0 + 5
                    i2 = i1 + nums
                    for i in range(i1, i2):
                        lspl = lines[i].split()
                        samp = []
                        for j in range(0, 2):
                            samp.append(float(lspl[j]))
                        ldos.append(samp)
                    l_pd = np.array(ldos).T
                    # apply
                    self.set_temperature(temp)
                    self.atom = { # atom definition
                        "Z" : Znum,
                        "symbol" : symb,
                        "mass" : mass
                    }
                    self.set_pdos(l_pd[0], l_pd[1], 0.001)

    def get_utsqr_ho(self, e, t):
        '''

        get_utsqr_ho

        Returns the mean squared displacement for the current
        atom and assuming a harmonic oscillator with Einstein
        energy e = hbar omega

        Parameters
        ----------

            e : float
                phonon energy in eV
            t : float
                temperature in K

        Returns
        -------

            float
                mean squared displacement in [A^2]

        '''
        ma = self.atom['mass'] * ec.PHYS_MASSU # mass in kg
        w = e / ec.PHYS_HBAREV # oscillation Einstein frequency in Hz
        return ho.usqrt(ma, w, t) * 1.E+20 # msd in [A^2]

    def get_utsqr(self, t, dEp=-1.0, verbose=0):
        '''

        get_utsqr

        Returns the mean squared displacement for the current
        atom and PDOS 
        '''
        uts = 0.0
        de = self.pdos["data"]["energy_step"] # use internal energy grid step
        if dEp > 0.0: de = dEp
        if verbose > 0: print("energy step:", de)
        pdos = self.get_pdos_pdf(de)
        n = len(pdos["energy"])
        if verbose > 0: print("n:", n)
        et = t * ec.PHYS_KB / ec.PHYS_QEL # thermal energy in eV
        if verbose > 0: print("Et:", et)
        ma = self.atom["mass"] # atomic mass in u
        s = np.zeros(n, dtype=np.float64)
        spdos = np.trapz(pdos["pdf"], x=pdos["energy"])
        if verbose > 0: print("total pdos:", spdos)
        pdos_thr = self.pthr * np.amax(pdos["pdf"])
        if n > 0:
            fm = 0.5E+20 * ec.PHYS_HBAREV * ec.PHYS_HBAR / (ma * ec.PHYS_MASSU)
            if verbose > 0: print("fm:", fm)
            for i in range(0, n):
                ct = 1.0
                ep = pdos["energy"][i]
                if (ep > 0.0 and pdos["pdf"][i] > pdos_thr):
                    if (et > 0.0):
                        ct = 1.0 / np.tanh(0.5 * ep / et)
                    s[i] = fm * pdos["pdf"][i] * ct / ep
            uts = np.trapz(s, x=pdos["energy"]) / spdos
        return uts
    

    def prepare_displ_mct(self):
        '''

        prepare_displ_mct

        Prepares random number generators for a displacement
        Monte-Carlo based on the PDOS at a given temperature.
        The result and functional objects are stored in the
        dictionary rngdist.

        This is a reduced version of prepare_displ_mc, which
        can be used with get_displt, achieving the same result
        as prepare_displ_mc with get_displ.
        prepare_displ_mc and get_displ may be deprecated at some
        point.

        Depends on the current setup of PDOS (set_pdos),
        temperature (set_temperature), and atom (set_atom).

        Prepares several instances of emilys.numerics.rngdist
        objects which will be used in a sequence to calculate
        random variates of atomic displacements.
        
        '''
        rd = self.rngdist
        # Store current PDOS info as distribution.
        # This is done because the rng works with an energy range
        # starting from E = 0, while the original PDOS in self.pdos
        # starts from E > 0.
        rd["pdos"] = { "dist" : self.get_pdos() }
        d_pdos = rd["pdos"]["dist"]
        # Init a discrete rng for the PDOS.
        # Use this to draw the phonon energy.
        rd["pdos"]["rng"] = rngdist(d_pdos["energy"], d_pdos["pdf"])
        
    
    def prepare_displ_mc(self):
        '''

        prepare_displ_mc

        Prepares random number generators for a displacement
        Monte-Carlo based on PDOS, Boltzmann and positional
        distribution functions. The result and functional
        objects are stored in the dictionary rngdist.

        Depends on the current setup of PDOS (set_pdos),
        temperature (set_temperature), and atom (set_atom).

        Prepares several instances of emilys.numerics.rngdist
        objects which will be used in a sequence to calculate
        random variates of atomic displacements.
        
        '''
        rd = self.rngdist
        # Store current PDOS info as distribution.
        # This is done because the rng works with an energy range
        # starting from E = 0, while the original PDOS in self.pdos
        # starts from E > 0.
        rd["pdos"] = { "dist" : self.get_pdos() }
        d_pdos = rd["pdos"]["dist"]
        # Init a discrete rng for the PDOS.
        rd["pdos"]["rng"] = rngdist(d_pdos["energy"], d_pdos["pdf"])
        #
        # store displacement scales for all phonon energies
        # this is needed to rescale the oscillator state displacement rng
        # which is working on a reference MSD of 1
        sca_u = d_pdos["energy"] * 0.0
        for i in range(0, len(d_pdos["energy"])):
            ep = d_pdos["energy"][i]
            if ep > 0:
                ma = self.atom["mass"] * ec.PHYS_MASSU
                sca_u[i] = 1.0E10 * ec.PHYS_HBAR / np.sqrt(ma * ep * ec.PHYS_QEL)
        rd["pdos"]["displacement_scale"] = sca_u
        #
        # Init discrete rngs for the initial states per phonon energy.
        # Use rngdist.rand_elem() to dice an energy index iep
        # in the list d_pdos["energy"].
        rd["pbol"] = {}
        nmax = 0
        for iep in range(0, len(d_pdos["energy"])): # loop over phonon energies
            s_iep = str(iep)
            ep = d_pdos["energy"][iep]
            if ep < 0.0001: continue # skip low phonon energies, Warning: This is a hard lower limit.
            rd["pbol"][s_iep] = { "energy" : ep, "dist" : self.get_pbol(ep) }
            db = rd["pbol"][s_iep]
            # Init a discrete rng for the Boltzmann distribution.
            db["rng"] = rngdist(db["dist"]["quantum_numbers"], db["dist"]["pdf"])
            # update the maximum quantum number to take into account
            nmax = max(nmax, db["dist"]["nmax"])
        #
        # Init continuous rngs for displacements per quantum number
        # of 1d harmonic oscillator states
        rd["displ"] = { "nmax" : nmax, "ngrid" : 1000 }
        for ni in range(0, nmax+1):
            s_ni = str(ni)
            rd["displ"][s_ni] = {}
            dd = rd["displ"][s_ni]
            dd["dist"] = ho.get_pdf(ni, rd["displ"]["ngrid"])
            dd["rng"] = rngdist(dd["dist"]["z"], dd["dist"]["pdf"], num_icdf=dd["dist"]["ngrid"] * 10)
        

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

    def prepare_feq(self):
        '''

        prepare_feq

        Prepares atom scattering factors on the standard q-grid.

        This function is called from class functions and
        should not be used elsewhere. It relies on a full
        setup of the object (grid and detector)

        Current implementation is for square grids.

        '''
        gtp = self.qgrid
        n = gtp["n"]
        dq = self.qgrid["dq"]
        l_q2ex = np.sum(np.meshgrid((gtp["l_qi"] * dq)**2, (gtp["l_qi"] * dq)**2), axis=0)
        l_qabs = np.sqrt(l_q2ex)
        gtp["l_feq"] = get_fen(l_qabs.flatten(), self.atom["data"]["q"], self.atom["data"]["fe"]).reshape(n,n).astype(np.double)


    def get_trstr0(self, ep, mx, nx, my, ny):
        '''

        get_trstr0

        Calculates the transition strength for a plane
        incident wave function on the q grid grid for
        the current detector.

        This function is called from class functions and
        should not be used elsewhere. It relies on a full
        setup of the object (grid and detector)

        '''
        # q-grid
        dq = self.qgrid["dq"]
        pfac = 1.0 * dq * dq # prefactor
        l_q = self.qgrid["l_q"] # q for the detector hash indices
        feq = self.qgrid["l_feq"] # scattering factors prepared on the q grid
        # detector
        dethash = self.det["hash"]["index"]
        detvals = self.det["hash"]["value"]
        nd = len(dethash) # number of detector pixels
        # get oscillator parameters
        wi = ep / ec.PHYS_HBAREV
        mat = self.atom["mass"] * ec.PHYS_MASSU # atom mass in kg
        uavg = self.atom["utsqr"]
        ui = ho.usqr0(mat, wi) * 1.E20 # in A^2
        # calculate Hnm factors for detector q
        iq = np.array(dethash, dtype=float).T # detector pixel indices
        uiy = np.unique(iq[0].astype(int)) # unique list for qiy
        uqy = l_q[uiy] # list of unique qy
        iuy = np.zeros((np.amax(uiy)+1), dtype=int)
        for i in range(0, len(uiy)): # generate look-up table for unique iy
            iuy[uiy[i]] = i
        uix = np.unique(iq[1].astype(int))
        uqx = l_q[uix] # unique qx
        iux = np.zeros((np.amax(uix)+1), dtype=int)
        for i in range(0, len(uix)):
            iux[uix[i]] = i
        ppy = ho.tsq_mod(uqy,ui,uavg,my,ny)
        ppx = ho.tsq_mod(uqx,ui,uavg,mx,nx)
        s = 0.0
        for i in range(i, nd): # loop over detector pixels
            p = dethash[i]
            sdet = detvals[i]
            ix = iux[p[1]]; iy = iuy[p[0]]
            ds = ppy[iy] * ppx[ix] * feq[p[0],p[1]] # calculate amplitude
            #print(i, p, [uqy[iy], uqx[ix]], [ppy[iy], ppx[ix]], ds)
            ds2 = ds.real**2 + ds.imag**2 # take mod square (inelastic diffraction pattern)
            s += (ds2 * sdet) # sum up with sensitivity
        return s * pfac
    
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
        pfac = 1.0 * dq * dq # * (ec.PHYS_HPL**2 / (2*np.pi * ec.EL_M0))**2 # constant prefactor (h^2 / (2 pi m_el))^2 dqx dqy
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
        detvals = self.det["hash"]["value"]
        j = numint_muh(l_qi, l_gsh[1], l_gsh[0], dethash, detvals, hx, hy, feq, sdet)
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

        Relativistic correction is not applied yet.

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
    
    def get_tpq_mod(self, ep, uavg, mx, nx, my, ny):
        '''

        get_tpq_mod

        Calculates an effective transition potential as a function of q
        for the given phonon energy ep and the transition quantum
        numbers mx -> nx, my -> ny and an effective average MSD uavg.

        The returned data is missing a prefactor
        h^2 / (2 pi m0)
        where m0 is the electron rest mass.

        Relativistic correction is not applied yet.

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
        hx = ho.tsq_mod(l_q, u02, uavg, mx, nx) # x mode transition factors <a_nx|H|a_mx>(q)
        hy = ho.tsq_mod(l_q, u02, uavg, my, ny) # y mode transition factors <a_ny|H|a_my>(q)
        h = np.outer(hy, hx) * gtp["l_feq"] # H(qy,qx) = Hx(qx) * Hy(qy) * fe(qy,qx)
        return h
    
    def get_grid_qsqr_ex(self, qx, qy):
        '''

        get_grid_qsqr_ex

        Returns a 2d grid of qx**2 + qy**2.

        '''
        qxx, qyy = np.meshgrid(qx, qy)
        return qxx**2 + qyy**2

    def get_tpq_mod_ex(self, ep, qx, qy, uavg, mx, nx, my, ny):
        '''

        get_tpq_mod_ex

        Calculates an effective transition potential as a function of q
        for the given phonon energy ep and the transition quantum
        numbers mx -> nx, my -> ny and an effective average MSD uavg.

        The returned data is missing a prefactor
        h^2 / (2 pi m0)
        where m0 is the electron rest mass.

        Relativistic correction is not applied yet.

        Returns an array h of values for a q-grid defined by arrays qx
        and qy, where the rows will be sorted according to qy and the
        items in each row (i.e. the columns) according to qx, such that
        h[i,j] if for a vector q = [qx[j], qy[i]].

        Requires the atomic scattering factor to be accessible,
        i.e. call the routine set_atom before using this function.

        The output is an array of type numpy.complex128

        '''
        # calculate the q grid
        ndim = np.array([len(qy), len(qx)])
        a_q = np.sqrt(self.get_grid_qsqr_ex(qx, qy))
        # get oscillator parameters
        w = ep / ec.PHYS_HBAREV # frequency in Hz
        mat = self.atom["mass"] * ec.PHYS_MASSU # atom mass in kg
        u02 = ho.usqr0(mat, w) * 1.E20 # ground state MSD in A^2
        # calculate transition potential -> h
        hx = ho.tsq_mod(qx, u02, uavg, mx, nx) # x mode transition factors <a_nx|H|a_mx>(q)
        hy = ho.tsq_mod(qy, u02, uavg, my, ny) # y mode transition factors <a_ny|H|a_my>(q)
        a_feq = get_fen(a_q.flatten(), self.atom["data"]["q"], self.atom["data"]["fe"]).reshape(ndim).astype(np.double)
        h = np.outer(hy, hx) * a_feq # H(qy,qx) = Hx(qx) * Hy(qy) * fe(qy,qx)
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
        dq = self.qgrid["dq"]
        a = np.array([1./dq, 1./dq])
        pfac = 1.0 # * (ec.PHYS_HPL**2 / (2*np.pi * ec.EL_M0))**2 # constant prefactor (h^2 / (2 pi m_el))^2 dqx dqy
        # ^^    need to clarify the units here
        # calculate transition potential -> h
        h = self.get_tpq(ep, mx, nx, my, ny)
        # calculate inelastic wave function (multiply h in real space)
        pinel = probe * prb.norm_ifft2(h, a) #* sca_ift # ... (missing I sigma ) ...
        pinelq = prb.norm_fft2(pinel, a) #* sca_ft # inelastic wavefunction in q-space
        # inelastic diffraction pattern
        difpat = pinelq.real**2 + pinelq.imag**2
        # ... integration over the detector area : sum(abs(psi)^2 * dqx * dqy)
        sdet = np.sum(difpat*l_det) / np.product(a)
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
        dq = self.qgrid["dq"]
        a = np.array([1./dq, 1./dq])
        nms = len(ep) # number of scattering events (assume all other input is likewise of that length)
        assert nms > 0, 'This requires list inputs of at least length 1.'
        pfac = 1.0 # * (ec.PHYS_HPL**2 / (2*np.pi * ec.EL_M0))**2 # constant prefactor (h^2 / (2 pi m_el))^2 dqx dqy
        sca_ift = 1.0 # dq * dq # iFT scaling factor 
        sca_ft = 1.0 # dq * dq # FT scaling factor 
        # ^^    need to clarify the units here
        # calculate transition potential -> h
        h = self.get_tpq(ep[0], mx[0], nx[0], my[0], ny[0])
        # calculate inelastic wave function after first transition (real-space)
        pinel = probe * prb.norm_ifft2(h, a) * sca_ift # ... (missing I sigma ) ...
        if (nms > 1):
            for ims in range(1, nms):
                # calculate transition potential -> h
                h = self.get_tpq(ep[ims], mx[ims], nx[ims], my[ims], ny[ims])
                pinel = (pinel * prb.norm_ifft2(h, a) * sca_ift) # ... apply next TP in real space
        # calculate inelastic wave function (multiply h in real space)
        pinelq = prb.norm_fft2(pinel, a) * sca_ft # inelastic wavefunction in q-space
        # inelastic diffraction pattern
        difpat = pinelq.real**2 + pinelq.imag**2
        # ... integration over the detector area : sum(abs(psi)^2 * dqx * dqy)
        sdet = np.sum(difpat*l_det) / np.product(a)
        return sdet * pfac

    def get_sdf(self, t, E0, E1, dE, Eres, method='msd', verbose=0):
        '''

        get_sdf

        Returns a spectral distribution function.

        Requires setup of atom and pdos.

        Parameters
        ----------
        t : float
            temperature in K
        E0 : float
            minimum energy loss considered in eV
        E1 : float
            maximum energy loss considered in eV
        dE : float
            step size of the spectrum in eV
        Eres : float
            energy resolution applied in eV
            make this larger than the internal energy grid step
            especially if dE is smaller than this step size
        method : str, default: "msd"
            calculation method:
                "msd" : using proportionality to mean squared displacements
                        in a harmonic oscillator model for small scattering angles,
                        single phonon excitations, and a plane incident wavefunction
                "tps" : using a transition potential approach for single-phonon
                        excitations, taking the detector function into account,
                        and a plane incident wavefunction,
                        requires additional setup of q-grid and detector

        Returns
        -------
        dict
            spectral distribution function data as dictionary

        '''
        pdos = self.get_pdos()
        Ep_max = np.amax(pdos["energy"])
        dEp = pdos["energy_step"]
        # get the energy loss range based on the PDOS grid, extend to gains
        # assuming single-phonon excitations only
        l_el0 = np.arange(-Ep_max, Ep_max + 0.5 * dEp, dEp)
        l_sdf0 = l_el0 * 0.0
        nel = len(l_el0)
        npdos = len(pdos["energy"])
        pdos_thr = self.pthr * np.amax(pdos["pdf"])
        self.atom["utsqr"] = self.get_utsqr(self.t)
        #
        if method == "msd": # simple MSD based method (good for small on-axis detectors)
            for i in range(0, npdos): # loop over phonon energy
                ep = pdos["energy"][i]
                if ep <= 0.0: continue # only positive phonon energies
                jl = int(np.round((Ep_max + ep)/ dEp)) # index of energy-loss
                jg = int(np.round((Ep_max - ep)/ dEp)) # index of energy-gain
                pdosi = pdos["pdf"][i] # pdos at ep
                if pdosi < pdos_thr: continue # ignore this contribution due to low probability
                uts = self.get_utsqr_ho(ep, t) # msd of the QHO at ep and t
                b = pbolz(-ep, t) # botzmann factor
                if (jl>=0 and jl<nel):
                    l_sdf0[jl] = pdosi * uts * b / (1.0 + b)
                if (jg>=0 and jg<nel):
                    l_sdf0[jg] = pdosi * uts / (1.0 + b)
            #
        elif method == "tps": # single-phonon transition potential method (OK also for off-axis)
            for i in range(0, npdos): # loop over phonon energy
                ep = pdos["energy"][i]
                if ep <= 0.0: continue # only positive phonon energies
                jl = int(np.round((Ep_max + ep)/ dEp)) # index of energy-loss
                jg = int(np.round((Ep_max - ep)/ dEp)) # index of energy-gain
                pdosi = pdos["pdf"][i] # pdos at ep
                if pdosi < pdos_thr: continue # ignore this contribution due to low probability
                w = ep / ec.PHYS_HBAREV # oscillator frequency
                nimax = nmaxt(ep, self.t, self.pthr) # get max. initial state quantum number to take into account
                if verbose > 0: print('(get_sdf): including initial states up to m = {:d} ...'.format(nimax))
                nrmtbd = (np.exp(ep/self.tev) - 1)**2 / np.exp(ep/self.tev) # normalization factor for the 2-d boltzmann distribution
                # boltzmann distribution threshold
                pb_thr = nrmtbd * pbolz(ep, self.t) * self.pthr # 2d ground state occupation time relative threshold
                if verbose > 0: print('(get_sdf): allowing 2d Boltzmann factors above {:.2f}% ...'.format(pb_thr*100.))
                # per phonon energy (to be weighted by pdos)
                for niy in range(0, nimax+1): # loop over initial states in the y dimension
                    eiy = ho.En(niy, w) / ec.PHYS_QEL # initial y state energy in eV
                    pby = pbolz(eiy, self.t) # initial y state Boltzmann distribution probability
                    for nix in range(0, nimax+1): # loop over initial states in the x dimension
                        eix = ho.En(nix, w) / ec.PHYS_QEL # initial x state energy in eV
                        pbx = pbolz(eix, self.t) # initial x state Boltzmann distribution probability
                        pb2 = nrmtbd * pbx * pby # normalized occupation of the initial state
                        if pb2 < pb_thr: continue # skip due to low initial state occupation probability
                        if verbose > 1: print('(get_sdf): * [{:.4f} eV] initial state [{:d},{:d}] (pbol = {:.2f}%) ...'.format(ep,nix,niy,pb2*100.))
                        if (jl>=0 and jl<nel): # energy loss transitions are in requested range
                            sx = self.get_trstr0(ep, nix, nix+1, niy, niy)
                            sy = self.get_trstr0(ep, nix, nix, niy, niy+1)
                            l_sdf0[jl] += pdosi * pb2 * (sx + sy) # add losses
                        if (jg>=0 and jg<nel): # energy gain transitions are in requested range
                            sx = 0.
                            sy = 0.
                            if (nix>0):
                                sx = self.get_trstr0(ep, nix, nix-1, niy, niy)
                            if (niy>0):
                                sy = self.get_trstr0(ep, nix, nix, niy, niy-1)
                            l_sdf0[jg] += pdosi * pb2 * (sx + sy) # add gains
        #
        # get convoluted spectrum on output range
        l_el1, l_sdf1 = get_spec_conv(l_el0, l_sdf0, E0, E1, dE, Eres)
        # normalization
        stot = np.trapz(l_sdf1, x=l_el1)
        l_sdf2 = l_sdf1 / stot
        #
        # generate dictionary return
        return {  "energy-loss" : l_el1,
                  "sdf" : l_sdf2,
                  "method" : method,
                  "temperature" : t,
                  "pdos" : pdos,
                  "atom" : self.atom["Z"],
                  "norm" : stot
                }

    def get_mul2d(self, energy_loss_range, verbose=0):
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

        Only single-phonon excitations are considered.

        Parameters
        ----------
            energy_loss_range : array-like, len=2, type float
                lower and upper bound of energy-losses in eV
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
                    # try all four possible single-phonon excitations
                    l_dnf = [[1, 0], [0, 1], [-1, 0], [0, -1]]
                    for dnf in l_dnf:
                        nfy = niy + dnf[0]
                        nfx = nix + dnf[0]
                        ntr = 0 # count for included number of transitions
                        if (nfy < 0 or nfx < 0): continue # skip negative qn
                        efy = ho.En(nfy, w) / ec.PHYS_QEL # final y state energy
                        efx = ho.En(nfx, w) / ec.PHYS_QEL # final x state energy
                        delE = efx - eix + efy - eiy # energy loss of the probing electron
                        idelE = int(np.rint(delE / dE)) # energy loss grid index
                        if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                            s = self.get_trstr(ep, nix, nfx, niy, nfy)
                            jdelE = idelE - imin
                            a_mu[jdelE] += (pdos * pb2 * s)
                            ntr += 1
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

    def get_spec_prb(self, probe, energy_loss_range, verbose=0):
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

        Only single-phonon excitations are considered.

        Parameters
        ----------
            probe : 2d numpy array, dtype=np.complex128
                probe wavefunction in real space
            energy_loss_range : array-like, len=2, type float
                lower and upper bound of energy-losses in eV
            verbose : int, default: 0
                verbosity level for text output
            
        Returns
        -------
            dict
                "data" : numpy.ndarray, num_dim=1, dtype=np.float64
                    phonon EELS spectrum
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
        #a_eels_sphon = a_eels * 0.0 # single phonon scattering & single inelastic scattering
        #a_eels_dmuls = a_eels * 0.0 # single inelastic scattering
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
                    # try all four possible single-phonon excitations
                    l_dnf = [[1, 0], [0, 1], [-1, 0], [0, -1]]
                    for dnf in l_dnf:
                        nfy = niy + dnf[0]
                        nfx = nix + dnf[0]
                        ntr = 0 # count for included number of transitions
                        if (nfy < 0 or nfx < 0): continue # skip negative qn
                        efy = ho.En(nfy, w) / ec.PHYS_QEL # final y state energy
                        efx = ho.En(nfx, w) / ec.PHYS_QEL # final x state energy
                        delE = efx - eix + efy - eiy # energy loss of the probing electron
                        idelE = int(np.rint(delE / dE)) # energy loss grid index
                        if (idelE >= imin) and (idelE <= imax): # energy loss is in the requested range
                            s = self.get_trstr_tp(probe, ep, nix, nfx, niy, nfy)
                            jdelE = idelE - imin
                            sw = (pdos * pb2 * s)
                            a_eels[jdelE] += sw
                            ntr += 1
                    #
                    ntr_total += ntr # sum total transitions
                    if verbose > 1:
                        print('(get_spec_prb): * transitions added: {:d}'.format(ntr))
                        print('(get_spec_prb): * total: {:.4E}'.format(np.sum(a_eels)))
                    #
                #
            #
        #
        if verbose > 0:
            print('(get_spec_prb): total number of transitions considered: {:d}'.format(ntr_total))
            print('(get_spec_prb): * total: {:.4E}'.format(np.sum(a_eels)))
        #
        # returning
        return {
            "data" : a_eels,
            "l_dE" : l_dE
        }
    
    def get_displt(self, num=1, num_enrg=1, verbose=0):
        """

        get_displt
        ---------

        Returns an array of random displacements for the current
        setup.
         
        Requires a previous call to prepare_displ_mct().

        Depends on the current setup of PDOS (set_pdos),
        temperature (set_temperature), and atom (set_atom).

        parameters
        ----------

        num : int, default: 1
            number of random displacements to generate
        num_enrg : int, default: 1
            number of consecutive displacements generated with
            the same phonon energy
        verbose : int, default: 0
            switch for text output

        returns
        -------

        numpy array, dtype=numpy.float64, shape=(num)
            an array of num random displacements in A
        
        """
        assert "pdos" in self.rngdist, 'no pdos setup'
        m = ec.PHYS_MASSU * self.atom["mass"] # atom mass in kg
        displ = np.zeros(num, dtype=np.float64) # init displacement array
        l_ev = self.rngdist["pdos"]["dist"]["energy"] # init energy list [eV]
        l_iep = self.rngdist["pdos"]["rng"].rand_elem(num) # get random phonon energy indices
        j = -1
        for i in range(0, num): # loop displacement draws
            if (i % num_enrg) == 0: # need a new phonon energy?
                j += 1 # advance in phonon energy list on every num_enrg-th pass
            iep = l_iep[j] # phonon energy index
            w = l_ev[iep] / ec.PHYS_HBAREV # phonon frequency
            usca = np.sqrt(ho.usqrt(m, w, self.t)) * 1.0E10 # rmsd in A
            displ[i] = usca * np.random.normal() # get displacement in A
            if verbose > 1:
                print("(get_displ): E = {:.4f} eV, u = {:.3e} A".format(l_ev[iep], displ[i]))
        return displ
    
    def get_displ(self, num=1, num_enrg=1, verbose=0):
        """

        get_displ
        ---------

        Returns an array of random displacements for the current
        setup. Requires a previous call to prepare_displ_mc().

        parameters
        ----------

        num : int, default: 1
            number of random displacements to generate
        num_enrg : int, default: 1
            number of consecutive displacements generated with
            the same phonon energy
        verbose : int, default: 0
            switch for text output

        returns
        -------

        numpy array, dtype=numpy.float64, shape=(num)
            an array of num random displacements
        
        """
        assert "displ" in self.rngdist, 'no rng setup'
        displ = np.zeros(num, dtype=np.float64) # init displacement array
        l_iep = self.rngdist["pdos"]["rng"].rand_elem(num) # get random phonon energy indices
        j = -1
        for i in range(0, num): # loop displacement draws
            if (i % num_enrg) == 0: # need a new phonon energy?
                j += 1 # advance in phonon energy list on every num_enrg-th pass
            iep = l_iep[j] # phonon energy index
            s_iep = str(iep) # ... as str to access the correct Boltzmann distribution
            sca = self.rngdist["pdos"]["displacement_scale"][iep] # scaling factor
            ni = self.rngdist["pbol"][s_iep]["rng"].rand_discrete()[0] # random initial state quantum number
            s_ni = str(ni) # ... as str to access the correct spatial distribution
            displ[i] = sca * self.rngdist["displ"][s_ni]["rng"].rand_continuum()[0] # get displacement
            if verbose > 1:
                print("(get_displ): iep = " + s_iep + ", ni = " + s_ni + ": u = {:.3e}".format(displ[i]))
        return displ
    
    def get_displ_einstein(self, ev, t, num=1):
        """

        Returns an array of random displacements for the current
        setup assuming an Einstein model with phonon energy ev
        and temperature t.

        Requires a previous setup of atom data (set_atom).
        
        """
        assert ev > 0.0, "Expecting positive energy input in eV."
        assert t >= 0.0, "Expecting non-negative temperature in K."
        ma = self.atom["mass"] * ec.PHYS_MASSU
        sca = 1.0E10 * ec.PHYS_HBAR / np.sqrt(ma * ev * ec.PHYS_QEL)
        tev = t * ec.PHYS_KB / ec.PHYS_QEL # thermal energy in eV
        displ = np.zeros(num, dtype=np.float64)
        #
        # setup rngs for the initial states from the Boltzmann distribution
        t_bk = self.t
        self.set_temperature(t)
        dn = self.get_pbol(ev)
        nmax = dn["nmax"]
        rng_ni = rngdist(dn["quantum_numbers"], dn["pdf"]) # setup initial state rng
        d_displ = {}
        ngrid = 1000
        for ni in range(0, nmax+1): # setup displacement rngs for all initial states
            s_ni = str(ni)
            d_displ[s_ni] = {}
            dd = d_displ[s_ni]
            dd["dist"] = ho.get_pdf(ni, ngrid)
            dd["rng"] = rngdist(dd["dist"]["z"], dd["dist"]["pdf"], num_icdf=dd["dist"]["ngrid"] * 10)
        for i in range(0, num):
            ni = rng_ni.rand_discrete()[0]
            s_ni = str(ni)
            displ[i] = sca * d_displ[s_ni]["rng"].rand_continuum()[0]
        self.set_temperature(t_bk)
        return displ