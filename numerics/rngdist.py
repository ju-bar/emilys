# -*- coding: utf-8 -*-
"""
Created on Mon Jul 3 14:27:00 2023
@author: ju-bar

Random number generation for user-defined probability distribution

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
from scipy import interpolate
# %%
class rngdist:
    '''
    
    class rngdist

    Generates random number for a user-defined probability
    distribution. The probability distribution is given on
    an equidistant grid.

    Members
    -------
        x : array like
            equidistant grid of variable values 
        pdf : array_like
            grid defining the probability distribution function
            for the grid x of values
        cdf : array like
            grid defining the cumulative probability distribution
            function for the grid x of values
    
    '''
    def __init__(self, x, pdf, num_icdf=0):
        n = len(x)
        assert n > 0, 'Expecting non zero length of input x.'
        assert len(pdf) == n, 'Expecting equal length of inputs x and pdf.'
        self.x = x
        self.dx = x[1] - x[0] # input step size
        s = np.sum(pdf) * self.dx
        self.pdf = pdf / s # store the normalized pdf
        self.cdf = pdf * 0.0
        scum = 0.0
        for i in range(0, n):
            scum += (self.pdf[i] * self.dx)
            self.cdf[i] = scum
        self.num_icdf = num_icdf
        if num_icdf > 1: # define an inverse cdf with this number of samples
            self.p_icdf = np.arange(0, num_icdf, dtype=np.float64) / (num_icdf - 1)
            ip1_cdf = interpolate.interp1d(x = self.cdf, y = self.x + 0.5*self.dx
                                           , kind='linear', copy=True, bounds_error=False
                                           , fill_value=(self.x[0],self.x[-1]))
            self.x_icdf = ip1_cdf(self.p_icdf)
            del ip1_cdf
            self.icdf = interpolate.interp1d(x = self.p_icdf, y = self.x_icdf
                                           , kind='linear', copy=True, bounds_error=False
                                           , fill_value=(self.x_icdf[0],self.x_icdf[-1]))

    def rand_elem(self, num=1):
        assert num > 0, 'Expecting num to be at least 1.'
        p = np.random.uniform(0., 1., num)
        elem = np.zeros(num, dtype=int)
        for i in range(0, num):
            elem[i] = np.argmax(self.cdf>=p[i])
        return elem
    
    def rand_discrete(self, num=1):
        elems = self.rand_elem(num)
        return self.x[elems]
    
    def rand_continuum(self, num=1):
        assert num > 0, 'Expecting num to be at least 1.'
        assert self.num_icdf > 1, 'Expecting icdf to be defined on init.'
        p = np.random.uniform(0., 1., num)
        return self.icdf(p)