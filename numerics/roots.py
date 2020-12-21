# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:17:00 2020
@author: ju-bar

Root finding functions

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
#
import numpy as np
#
def root_poly_2(y, a, b, c):
    """

    Returns the two roots of a 2nd order (quadratic) equation
    y = a * x**2 + b * x + c
    where a != 0.

    Parameters
    ----------
        y : float
            left side of the equation
        a : float
            2nd order coefficient
        b : float
            1st order coefficient
        c : float
            0th order coefficient

    Returns
    -------
        array of two numbers (possibly complex)

    """
    assert np.abs(a) > 0, 'this is for non-zero second order terms only'
    q = (c - y) / a
    p = b / a
    ph = p / 2
    pd = ph**2 - q
    x0 = -ph - pd**0.5
    x1 = -ph + pd**0.5
    return [x0, x1]
#
def root_poly_3(y, a, b, c, d):
    """

    Returns the three roots of a 3rd order (cubic) equation
    y = a * x**3 + b * x**2 + c * x + d
    where a != 0.

    Parameters
    ----------
        y : float
            left side of the equation
        a : float
            3rd order coefficient
        b : float
            2nd order coefficient
        c : float
            1st order coefficient
        d : float
            0th order coefficient

    Returns
    -------
        array of three numbers (possibly complex)

    Remarks
    -------
        (investigate why in some cases, the solutions are not correct)

    """
    assert np.abs(a) > 0, 'this is for non-zero cubic terms only'
    zeta1 = np.exp(2j * np.pi / 3) # cubic root of unity with zeta1**2 = imag(zeta1) = zeta2
    zeta2 = np.exp(-2j * np.pi / 3) # the other cubic root of unity
    dp = d - y # transform to equation with zero right side
    # transform to equation with 1 in front of x**3
    ap = b / a
    bp = c / a
    cp = dp / a
    # 0 = x**3 + a' * x**2 + b' * x + c'
    P = ap**2 - 3 * bp # = a'**2 - 2 * b'
    S = 2 * ap**3 - 9 * ap * bp + 27 * cp # = 2 * a'**3 - 9 * a' * b' + 27 * c'
    print('S = {:.4f}'.format(S))
    print('P = {:.4f}'.format(P))
    s13, s23 = root_poly_2(0, 1, S, P**3) # solves 0 = z**2 + S + P**3
    print('z1 = {:.4f}'.format(s13))
    print('z2 = {:.4f}'.format(s23))
    s0 = -ap # -a'
    s1 = s13**(1/3) # z1**(1/3)
    s2 = s23**(1/3) # z2**(1/3)
    x0 = (s0 + s1 + s2) / 3
    x1 = (s0 + zeta2 * s1 + zeta1 * s2) / 3
    x2 = (s0 + zeta1 * s1 + zeta2 * s2) / 3
    return [x0, x1, x2]