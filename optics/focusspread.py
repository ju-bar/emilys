# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:16:00 2020
@author: ju-bar

Functions related to optical focus spread

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
from numba import jit # include compilation support
import numpy as np
from emilys.image.polar import polar_transform
from scipy.optimize import curve_fit
# %%
def dist_gaussian(x, delta):
    '''

    Calculates values of a Gaussian focus spread distribution with
    focus spread 1/e-half-width delta.

    Parameters:
        x : float
            defocus value
        delta : float
            1/e-half-width of the focal distribution

    Returns:
        float

    '''
    return np.exp(-x**2 / delta**2) / (np.sqrt(np.pi) * np.abs(delta))

# %%
def info_limit(delta, lamb):
    '''

    Calculates the information limit for a Gaussian focus-spread with
    1/e half-width delta and for electrons of wavelength lamb.

    Parameters:
        delta : float
            focus spread 1/e-half-width
        lamb : float
            electron wave-length

    Returns:
        float : info-limit (same unit as delta and lamb)

    '''
    return ((np.pi*lamb*delta)**2 / 8.)**0.25
# %%
def delta_of_kappa(kappa, g, t, lamb):
    '''
    Calculate the focus-spread parameter from the circ_lf model
    parameter kappa for a given reciprocal space circle radius g
    beam tilt magnitude t and electron wavelength lamb
    '''
    return np.sqrt(0.5 * kappa) / (np.pi * lamb * g * t)
# %%
def kappa_of_delta(delta, g, t, lamb):
    '''
    Calculate the kappa parameter for the circ_lf model from
    focus spread delta for a given reciprocal space circle radius g
    beam tilt magnitude t and electron wavelength lamb
    '''
    return 2.0 * (np.pi * lamb * g * t * delta)**2

# %%
def circ_lf(phi, phi_t, g, t, c0, c1, kappa):
    '''
    
    Calculates a value of the low-frequency diffractogram component
    under tilted beam illumination in polar notation for points (g, phi) 
    in the diffractogram, beam tilts (t, phi_t), assuming an additional
    base line c0, a modulation amplitude c1 and focus spread dependent
    parameter kappa (see delta_of_kappa).
    The model parameters c0, c1 and kappa are usually not known and
    can be determined from values of the low-frequency content scanned
    along circles around the origin.

    Parameters:
        phi : float
            diffractogram point azimuth in radians
        phi_t : float
            beam tilt azimuth in radians
        g : float
            circle spatial frequency
        t : float
            beam tilt magnitude in units of a spatial frequency
        c0 : float
            base line value
        c1 : float
            contrast amplitude
        kappa : float
            width parameter depending on focus spread
    
    Returns: float

    '''
    return c0 + c1 * np.exp(-kappa * np.cos(phi - phi_t)**2) * np.cosh(kappa * g / t * np.cos(phi - phi_t))
# %%
def circ_mod(xdata, xt, c0, c1, kap):
    '''
    Circular model of the low-frequency components with tilted beam illumintion.
    
    Parameters:
        xdata : numpy array of dimension 2
            xdata[0,:] = x values to calculate for (azimuth)
            xdata[1,0] = g value of the fit
            xdata[2,0] = t value of the fit
        xt : float
            tilt azimuth
        c0 : float
            base line
        c1 : float
            amplitude
        kap : float
            target parameter kap = 2.0 * (np.pi * lamb * g * t * delta)**2
    '''
    x, g, t = xdata[0,:], xdata[1,:], xdata[2,:]
    return circ_lf(x, xt, g[0], t[0], c0, c1, kap)
# %%
def measure_fs_lf(lf_dif, samp_q, q_rng, tilt, lamb):
    '''

    Measure the focus spread assuming a Gaussian focal distribution from
    the low-frequency component of a diffractogram recorded under tilted
    beam illumination with a thin amorphous sample and significant defocus.
    The analysis will be done on rings in the pattern with radius q_min
    to q_max. A square size input lf_dif is assumed with isotropic sampling.

    Parameters:
        lf_dif : numpy.array of 2 dimensions (square size assumed)
            low-frequency component of the diffractogram with Thon-rings
            effectively filtered out
        samp_q : float
            sampling rate of the diffractogram data (e.g. 1/nm / pixel)
            along both grid directions (isotropic sampling assumed)
        q_rng : array of length 2 (q_min, q_max)
            Range of spatial frequencies used in the analysis in the same
            unit as samp_q (e.g. 1/nm). Check that there is no Thon-ring
            modulation left in lf_dif over this range.
        tilt : array of length 2 (tilt_x, tilt_y)
            beam tilt applied for the measurement in the same unit as samp_q
            and q_rng (e.g. 1/nm).
        lamb : float
            electron wavelength in the reciprocal unit of samp_q (e.g. nm).

    Returns:
        delta, err_delta : 2 floats
            focus spread and its error estimate

    Remarks:
        1) designed to primarily work with nm and 1/nm units
        2) input data dimension should be at least 256 x 256

    '''
    nd = lf_dif.shape
    assert len(nd)==2, 'parameter 1 must be a 2d array'
    ndim = nd[0]
    assert (ndim > 32 and nd[1]==ndim), 'parameter 1 must be of some square size (>32)'
    assert samp_q > 0., 'parameter 2 must be larger than zero'
    ndim2 = ndim >> 1
    q_max = ndim2 * samp_q
    assert len(q_rng)==2, 'parameter 3 must have length 2'
    assert (q_rng[1] < q_max and q_rng[0] <= q_rng[1]), 'parameter 3 is not a valid range compared to parameter 1 and 2'
    assert len(tilt)==2, 'parameter 4 must have length 2'
    t_mod = np.sqrt(tilt[0]**2 + tilt[1]**2)
    assert t_mod > 0., 'parameter 4 is a vector of zero length'
    t_phi = np.arctan2(tilt[1],tilt[0])
    assert lamb > 0., 'parameter 5 is not a valid electron wavelength'
    d_q = q_rng[1] - q_rng[0]
    nrad = 1 + (int(d_q / samp_q + 0.5) >> 1) # number of rings from (range / sampling / 2)
    nphi = 180 # 180 samples along the azimuth
    porg = np.array([ndim2,ndim2]).astype(float) # origin of the pattern data in lf_dif
    prng = q_rng / samp_q # radial range in pixels
    arng = np.array([0.,2*np.pi]) # full azimuth range
    apol = polar_transform(lf_dif, nrad, nphi, porg, prng, arng) # get polar transform (could also use polar_resample)
    # --- fit model to all azimuthal curves ---
    xvals = np.zeros((3,nphi)) # prepare x-array
    xvals[0,:] = np.arange(0,nphi) * 2.0 * np.pi / nphi # set the azimuth values as x
    xvals[2,0] = t_mod # tilt modulus (same for all rings)
    qvals = np.arange(0,nrad) * (q_rng[1] - q_rng[0]) / nrad + q_rng[0] # set the radial (q) values
    lprm = np.zeros((nrad, 4)) # prepare array to store fit results for all rings
    lcov = np.zeros((nrad, 4, 4)) # prepare array to store covariance matrix for all rings
    ldel = np.zeros((nrad, 2)) # prepare array storing focus spread and its error for each ring
    for irow in range(0, nrad): # loop over all rings
        yvals = apol[irow] # get values on current ring
        ymin = np.amin(yvals) # minimum
        yamp = np.amax(yvals) - ymin # amplitude
        kap0 = kappa_of_delta(4., qvals[irow], t_mod, lamb) # kappa preset
        xvals[1,0] = qvals[irow] # q of ring
        popt, pcov = curve_fit(circ_mod, xvals, yvals, p0=np.array([t_phi, ymin, yamp, kap0])) # fit from presets
        lprm[irow,:] = popt # store best fitting parameters
        lcov[irow,:,:] = pcov # store covariance matrix
        delta = delta_of_kappa(popt[3], qvals[irow], t_mod, lamb) # calculate focus spread
        ldel[irow,0] = delta # store focus spread
        ldel[irow,1] = 0.25 * delta**2 * pcov[3,3] / popt[3]**2 # variance propagation from kappa
    m_phi = np.mean(lprm[:,0]) # mean tilt orientation
    v_phi = np.mean(lcov[:,0,0]) # tilt orientation variance
    if (np.abs(m_phi - t_phi) > 2.*np.sqrt(v_phi)):
        print('Detected tilt orientation:', m_phi, 'rad')
        print('   input tilt orientation:', t_phi, 'rad')
    m_del = np.mean(ldel[:,0]) # mean delta
    v_del_in = np.mean(ldel[:,1]) # mean variance from fits
    v_del_ex = np.var(ldel[:,0]) # variance over rings
    v_del = max(v_del_in, v_del_ex) # use larger variance
    s_del = np.sqrt(v_del)
    return m_del, s_del, qvals, ldel # return mean and std. deviation for delta and the g-resolved measurement