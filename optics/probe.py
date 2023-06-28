# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 16:15:00 2023
@author: ju-bar

Implementations of calculating and manipulating
electron probe functions

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3


On box-normalizations of probes:

Define the probe on a 2-dimensional grid of shape=(ny,nx) with
size a = (ay, ax). This defines the Area = ax * ay, the real-space
step size dr = (dy, dx) = (ay / ny, ax / nx), the q-space step
size dq = (dqy, dqx) = (1/ay, 1/ax).

The probe psi in real space is normalized by
Area = np.sum(psi.real**2 + psi.imag**2) * np.dot(dr,dr)
     = np.sum(psi.real**2 + psi.imag**2) * ay * ax / (ny * nx)
as implemented in r_box_norm.

The probe psi in Fourier space is normalized by
Area = np.sum(psi.real**2 + psi.imag**2) * np.dot(dq,dq)
     = np.sum(psi.real**2 + psi.imag**2) / (ay * ax)
as implemented in q_box_norm.

In order to preserve these norms under numpy fft2 operations, the
following scaling needs to be applied for forward transforms
numpy.fft.fft2 (r -> q):
psi_q = numpy.fft.fft2(psi_r) * dr
      = numpy.fft.fft2(psi_r) * (ay * ax) / (ny * nx),
as implemented in norm_fft2,
and for backwards transforms
numpy.fft.ifft2 (q -> r):
psi_r = numpy.fft.ifft2(psi_r) * dq * (ny * nx)
      = numpy.fft.ifft2(psi_r) (ny * nx) / (ay * ax),
as implemented in norm_ifft2.

"""

import numpy as np
from numba import njit
from emilys.optics.aperture import aperture

@njit
def get_probe_q(a, dq, q0, qa):
    ndim = np.array(a.shape, dtype=np.int32)
    if np.abs(qa) > 1.E-10:
        ndim2 = ndim >> 1
        lq0 = np.array([q0[1], q0[0]], dtype=np.float64)
        aq1 = dq[1] * (((np.arange(0, ndim[1]) + ndim2[1]) % ndim[1]) - ndim2[1])
        aq2 = dq[0] * (((np.arange(0, ndim[0]) + ndim2[0]) % ndim[0]) - ndim2[0])
        s = 0.0
        smt = np.float64(0.5 * (dq[0]+dq[1]))
        for i in range(0, ndim[0]):
            for j in range(0, ndim[1]):
                vq = np.array([aq1[j],aq2[i]], dtype=np.float64)
                v = aperture(vq, lq0, np.float64(qa), smt)
                a[i,j] = v
                s += v*v
        a[:,:] = a[:,:] / np.sqrt(s)
    else:
        a[:,:] = 0.0
        a[0,0] = 1.0 + 0.0J

def get_probe(a, dq, q0, qa):
    ndim = np.array(a.shape, dtype=np.int32)
    b = np.zeros(ndim, dtype=np.complex128)
    get_probe_q(b, dq, q0, qa)
    a[:,:] = np.fft.ifft2(b) * np.sqrt(ndim[0] * ndim[1])
    del(b)

@njit
def shift_probe_q(a_q_shift, a_q, dq, dr):
    '''
    shift_probe_q

    Modifies the input probe a_q, given in q-space, by applying
    phase shifts to realize a real-space shift of dr.

    njit compiled

    Parameters
    ----------
        a_q_shift : numpy array, shape=(ny,nx), dtype=complex
            output array of probe Fourier coefficients
        a_q  : numpy array, shape=(ny,nx), dtype=complex
            input array of probe Fourier coefficients
        dq : numpy array, shape(2), dtype=float
            q-space step size (y,x) in 1/A per pixel
        dr : numpy array, shape(2), dtype=float
            real-space shift (y,x) to apply in A
    Returns
    -------
        None
    '''
    ndim = np.array(a_q.shape, dtype=np.int32)
    ndim2 = ndim >> 1
    pq1 = -2.0J * np.pi * dr[1] * dq[1] * (((np.arange(0, ndim[1]) + ndim2[1]) % ndim[1]) - ndim2[1])
    pq0 = -2.0J * np.pi * dr[0] * dq[0] * (((np.arange(0, ndim[0]) + ndim2[0]) % ndim[0]) - ndim2[0])
    for i in range(0, ndim[0]):
        for j in range(0, ndim[1]):
            a_q_shift[i,j] = a_q[i,j] * np.exp(pq0[i] + pq1[j])

# function calculating the total norm in real-space
def r_box_norm(psi, a):
    '''
    y = (sum_i,j psi_i,j * dx*dy) = (sum_i,j psi_i,j) * A / (nx * ny)
    psi_i,j = psi(x, y)
    A = nx*dx * ny*dy
    (dy, dx) = grid step size
    (ny, nx) = grid size

    Parameters
    ----------
        psi : numpy array, shape=(ny, nx), dytpe=complex
            wave function
        a : numpy array, shape=(2), dtype=float-type
            box size (y,x)

    Returns
    -------
        norm : float = y
    '''
    ndim = np.array(psi.shape)
    nfac = np.product(a) / np.product(ndim)
    return np.sum(psi.real**2 + psi.imag**2) * nfac

# function calculating the total norm in q-space
def q_box_norm(psi, a):
    '''
    y = (sum_i,j psi_i,j * dqy*dqy) = (sum_i,j psi_i,j) / A
    psi_i,j = psi(qx, qy)
    A = nx*dx * ny*dy = 1 / (dqx*dqy)
    (dqy, dqx) = grid step size in reciprocal space
    (dy, dx) = grid step size in real space
    (ny, nx) = grid size

    Parameters
    ----------
        psi : numpy array, shape=(ny, nx), dytpe=complex
            wave function
        a : numpy array, shape=(2), dtype=float-type
            box size (y,x)

    Returns
    -------
        norm : float = y
    '''
    nfac = 1.0 / np.product(a)
    return np.sum(psi.real**2 + psi.imag**2) * nfac

def norm_fft2(psi, a):
    '''
    Norm preserving 2-d FFT

    Parameters
    ----------
        psi : numpy array, shape=(ny, nx), dytpe=complex
            wave function in real space
        a : numpy array, shape=(2), dtype=float-type
            box size (y,x)

    Returns
    -------
        numpy array, shape=(ny, nx), dytpe=complex
            wave function in Fourier space
    '''
    nfac = np.product(a) / np.product(np.array(psi.shape))
    return np.fft.fft2(psi) * nfac

def norm_ifft2(psi, a):
    '''
    Norm preserving 2-d inverse FFT

    Parameters
    ----------
        psi : numpy array, shape=(ny, nx), dytpe=complex
            wave function in Fourier space
        a : numpy array, shape=(2), dtype=float-type
            box size (y,x)

    Returns
    -------
        numpy array, shape=(ny, nx), dytpe=complex
            wave function in real space
    '''
    nfac = np.product(np.array(psi.shape)) / np.product(a)
    return np.fft.ifft2(psi) * nfac