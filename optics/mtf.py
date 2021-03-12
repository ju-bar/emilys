# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 08:42:00 2019
@author: ju-bar

Calculations and data handling related to the Modulation Transfer Function (MTF)
pixelated imaging detectors.


This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""

from numba import jit # include compilation support
import numpy as np # include numeric functions
from scipy.optimize import curve_fit

def loadmtf(sfilename):
    """

    Loads a list of values from a text file and returns a numpy array
    which is a list of tuples (frequency,mtf(frequency)).

    Parameters:
        filename: string
            The name of the file to load MTF data from

    Remarks:
        The file structure is expected to contain a number in the
        first row, specifying, which of the values following below
        corresponds to the Nyquist frequency (0.5) of the detector.
        If the detector has N pixels per row, this should be the
        item N/2 + 1.
        The routine attempts reading values beyond Nyquist, if present.

    """
    file_handle = None
    opened = False
    try:
        file_handle = open(sfilename, 'r') # try open the file
    except (OSError,IOError) as esc:
        print("Error: loadmtf failed:",format(esc))
    else:
        opened = True
        str_lines = file_handle.readlines()
    finally:
        if file_handle: file_handle.close()
        if not opened: return np.array([])

    nlines = len(str_lines)
    if nlines > 0:
        inyq = int(str_lines[0])-1
        inum = 2*inyq
        print("- number of mtf values: ", nlines-1)
        print("- Nyquist index: ", inyq-1)
        ares = np.zeros((nlines-1,2))
        for i in range(1,nlines):
            ares[i-1,0] = float(i-1)/inum
            ares[i-1,1] = float(str_lines[i])
        return ares
    else:
        return np.array([])

def modelmtf_func(x, d1, d2, mix):
    """

    MTF model function

    y = (1.-mix) * np.exp(-x/d1) + mix * np.exp(-0.5 * x**2 / d2**2)

    """
    return (1.-mix) * np.exp(-x/d1) + mix * np.exp(-0.5 * x**2 / d2**2)

def modelmtf(ndim, d1, d2, mix):
    """

    This calculates radial MTF model values on a grid of 1+ndim/2 points
    for x = 0 ... 0.5 according to
    y = (1.-mix) * np.exp(-x/d1) + mix * np.exp(-0.5 * x**2 / d2**2)

    Parameters
    ----------
        ndim : int
            number of image rows and image columns assuming a square image
        d1 : float
            exponential decay parameter
        d2 : float
            gaussian decay parameter
        mix : float
            mix between exponential and gaussian
            keep this between 0 and 1

    """
    ndim2 = ndim >> 1
    lx = np.arange(0, ndim2+1) / ndim
    ly = modelmtf_func(lx, d1, d2, mix)
    return np.array([lx, ly]).T

def getmtfkernel(lmtf, ndim, fsca=1.):
    """

    Calculates an MTF kernel array (numpy.array) for a given
    list of radial MTF values, frequency scale and array dimensions

    Parameters:
        lmtf: np.array(shape=(m,2))
            list of m tuples (frequency, mtf value)
        ndim: np.array(2)
            2D image (kernel) dimensions [ny, nx]
        fsca: float (default = 1.)
            frequency scale corresponding to the ratio of
            sampling rates (target / data) or (experiment / simulation)
            The parameter accepts an array of length 2 defining
            different scales for each dimension [fscay, fscax].

    Remarks:
        The MTF values are assumed to be without pixelation coefficients
        and sampled on an equidistant grid. The frequency sampling rate
        used is read from lmtf[1,0].
        MTF data is extrapolated by an exponential decay from the last
        third of values if provided data has only frequencies below 0.75.
        Pixelation coefficients are multiplied corresponding to the
        grid size defined by ndim.

    """
    # define local functions for tail fitting
    def texp(x, a, b, c): # exponential tail function
        return a * np.exp(-b * x) + c
    if ndim.size != 2: return np.array([])
    nmtf = lmtf.shape # number of available mtf data
    if len(nmtf) != 2: return np.array([])
    if nmtf[0] < 10: return np.array([])
    lfsca = np.array([1.,1.])
    if (np.size(fsca) > 1): # different scale for x and y
        lfsca = np.array(fsca.flatten()[0:2])
    else: # same scale for x and y
        lfsca[0] = np.array([fsca]).flatten()[0]
        lfsca[1] = lfsca[0]
    # initialize helper variables and arrays
    dfmtf = lmtf[1,0] # mtf frequency sampling
    fy = np.fft.fftfreq(ndim[0]) # y-frequency indices
    fx = np.fft.fftfreq(ndim[1]) # x-frequency indices
    amtf = np.zeros((ndim[0],ndim[1])) # prepare mtf array
    lmtf1 = lmtf # initialized used mtf data
    # mtf span check
    if lmtf[int(nmtf[0])-1,0] / np.max(lfsca) < 0.75: # do we need to extrapolate beyond the given mtf data ?
        # yes, extrapolate to f = 1.0
        lmtft = lmtf.T # get transposed data
        # fit to the last 1/3 values
        nfitrng = np.array([2*int(nmtf[0]/3),int(nmtf[0])-1]) # set fit index range
        fdata = lmtf[nfitrng[0]:nfitrng[1]].T # get data for fitting
        popt = curve_fit(texp, fdata[0], fdata[1]) # fit
        # extrapolate frequency range to 1 from the fitted exponential
        n1 = 1 + int(np.max(lfsca)/dfmtf + 0.5) # determine index range for frequencies up to 1
        # print('- extrapolation to f =', np.max(lfsca), ', index =', n1)
        fextr = np.arange(nfitrng[1]+1,n1) * dfmtf # prepare frequency array
        mtfextr = texp(fextr, *popt[0]) # calculate tail values
        lmtf1 = np.array([np.append(lmtft[0],fextr),np.append(lmtft[1],mtfextr)]).T # append and transpose back
        nmtf = lmtf1.shape # update number of available mtf data
    # mtf grid calculation
    for j in range(0,ndim[0]): # loop rows (v)
        fy2 = (lfsca[0] * fy[j])**2 # squared row frequency
        scy = np.sinc(fy[j]) # row sinc ! Sinc is for the given sampling, not affected by lfsca
        for i in range(0,ndim[1]): # loop columns (h)
            fm2 = (lfsca[1] * fx[i])**2 + fy2 # frequency modulus squared
            scx = np.sinc(fx[i]) # column sinc
            fm = np.sqrt(fm2) # scaled frequency modulus
            # prepare linear interpolation on radial mtf list
            km = fm/dfmtf # mtf array index corresponding to fm
            k0 = int(km) # lower array index
            fk = km - k0 # fraction to lower array index
            k1 = min(k0 + 1,nmtf[0]) # limited upper array index
            amtf[j,i] = scy * scx * ( (1. - fk) * lmtf1[k0,1] + fk * lmtf1[k1,1] ) # linear interpolation and sinc product
    return amtf