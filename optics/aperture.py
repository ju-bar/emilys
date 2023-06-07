# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 12:04:00 2019
@author: ju-bar

Calculation of 2-dimensional apertures

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
# %%
from numba import njit, float64 # include compilation support
import numpy as np # include numeric functions
# %%
@njit(float64(float64[:],float64[:],float64,float64))
def aperture(q=np.array([0.,0.]), q0=np.array([0.,0.]), qlim=1., qsmt=0.):
    """

    Calculates transmission values for a round smooth aperture in 2D.
    (numba compiled)

    Parameters:
        q : np.array[shape=(2)]
            query point
        q0 : np.array[shape=(2)]
            aperture center position
        qlim : float
            aperture radius
        qsmt : float
            edge smoothness
    """
    qs = np.abs(qsmt)
    ql = np.abs(qlim)
    if ql > 0.: # the aperture is open
        dq = q - q0
        dqm2 = np.dot(dq,dq)
        if dqm2 > 0.: # the query point is away from the aperture center
            dqe = dq / np.sqrt(dqm2) # unit vector pointing from aperture center to query point
            dql = dqe * ql # vector from aperture center to its edge in direction of query point, if the aperture is not round, this would need to be modified
            if qs > 0.: # the edge is smooth
                dqd = dq - dql # vector from aperture edge to query point (along the connection to the aperture center)
                arg = np.pi*np.dot(dqd,dqe) / qs # distance from the edge rescaled to Pi/qsmt
                return 0.5*(1. - np.tanh(arg)) # sigmoid edge transition from 1 to 0 
            else: # the edge is sharp
                if np.dot(dql,dql) > dqm2: # the query point is inside the aperture radius
                    return 1.
                else: # the query point is outside the aperture radius
                    return 0.
        else: # the query point is in the aperture center
            return 1.
    else: # the aperture is closed
        return 0.
    return 0. # default exit
# %%
@njit
def aperture_a(q=np.array([0.,0.]), q0=np.array([0.,0.]), qlim=1., qsmt=0., qa=0., qp=0.):
    """

    Calculates transmission values for a smooth aperture in 2D with 2-fold edge distortion
    (numba compiled)

    Parameters:
        q : np.array[shape=(2)]
            query point
        q0 : np.array[shape=(2)]
            aperture center position
        qlim : float
            aperture radius
        qsmt : float
            edge smoothness
        qa : float
            magnitude of 2-fold distortion
        qp : float
            direction of positive axis of 2-fold dist. [-Pi, Pi[
            
    """
    if (qlim > 0.):
        a1x = qa * np.cos( 2.* qp ) / qlim # rel. 2-fold distortion x component
        a1y = qa * np.sin( 2.* qp ) / qlim # rel. 2-fold distortion y component
        adet = 1. + (qa/qlim)**2
        if (adet == 0.0):
            return 0.0 # closed aperture by line distortion, exit
        radet = 1. / adet
        dq = q - q0 # distance vector to center
        dqx1 = ((1. - a1x)*dq[0] - a1y*dq[1])*radet # back project dqx to undistorted plane
        dqy1 = (-a1y*dq[0] + (1. + a1x)*dq[1])*radet # back project dqy to undistorted plane
        dq2 = dqx1 * dqx1 + dqy1 * dqy1
        dqm = np.sqrt(dq2) # undistorted distance from aperture center
        dqs = abs(qsmt) # absolute smoothness
        if (dqm > 0.0): # handle default case (beam is somewhere in aperture area)
            if (dqs > 0.0): # calculate smoothed aperture
                darg = np.pi * (dqm - qlim) / dqs
                return (1. - np.tanh(darg))*0.5 # aperture value
            else: # sharp aperture
                if (dqm < qlim):
                    return 1.0 # point is in the aperture
        else: # handle on-axis case
            return 1.0 # ... always transmit this beam
    else:
        return 0.
    return 0.

# %%
@njit
def aperture_dist3(q=np.array([0.,0.]), q0=np.array([0.,0.]), qlim=1., qsmt=0., qdist=np.array([])):
    """

    Calculates transmission values for a smooth aperture in 2D with edge distortions up to 3rd order.
    (numba compiled)

    Parameters:
        q : np.array[shape=(2)]
            query point
        q0 : np.array[shape=(2)]
            aperture center position
        qlim : float
            aperture radius
        qsmt : float
            edge smoothness
        qdist : np.array[shape=(na)]
            list of edge distortion parameters, length = na
            0,1 : two-fold distortion x, y
            2,3 : three-fold distortion x, y
            4,5 : four-fold distortion x, y
    """
    na = qdist.size # number of distortion parameters
    #print("dbg aperture_dist3: distortion parameters: na = ",na)
    qs = np.abs(qsmt) # strength of edge smoothness
    ql = np.abs(qlim) # aperture radius (must be positive)
    if ql > 0.: # the aperture is open
        dq = q - q0 # vector from aperture center to query point
        dqm2 = np.dot(dq,dq) # squared distance of query point from center
        if dqm2 > 0.: # the query point is away from the aperture center
            dqe = dq / np.sqrt(dqm2) # unit vector pointing from aperture center to query point [cos(p),sin(p)]
            dql = dqe * ql # vector from aperture center to its edge in direction of query point
            dql3 = dql
            if na > 0: # there are edge distortion parameters
                qx1 = dqe[0] # cos(p)
                qy1 = dqe[1] # sin(p)
                qx2 = qx1*qx1 # cos(p)**2
                qx3 = qx2*qx1 # cos(p)**3
                qy2 = qy1*qy1 # sin(p)**2
                qy3 = qy2*qy1 # sin(p)**3
                lbd3x = np.array([qx1,qy1,qx2-qy2,-2*qx1*qy1,qx3-3*qx1*qy2,3*qx2*qy1-qy3]) # aberration x basis
                lbd3y = np.array([-qy1,qx1,-2*qx1*qy1,qy2-qx2,qy3-3*qx2*qy1,qx3-3*qx1*qy2]) # aberration y basis
                for l in range(0, na):
                    #print("dbg aperture_dist3: l=",l,", qdist[l]=",qdist[l])
                    #print("dbg aperture_dist3: bqx(l)=",lbd3x[l],", bqy(l)=",lbd3y[l])
                    dql3 = dql3 + np.array([lbd3x[l]*qdist[l],lbd3y[l]*qdist[l]])
            if qs > 0.: # the edge is smooth
                dqd = dq - dql3 # vector from aperture edge to query point (along the connection to the aperture center)
                arg = np.pi*np.dot(dqd,dqe) / qs # distance from the edge rescaled to Pi/qsmt
                return 0.5*(1. - np.tanh(arg)) # sigmoid edge transition from 1 to 0 
            else: # the edge is sharp
                if np.dot(dql3,dql3) > dqm2: # the query point is inside the aperture radius
                    return 1.
                else: # the query point is outside the aperture radius
                    return 0.
        else: # the query point is in the aperture center
            return 1.
    else: # the aperture is closed
        return 0.
    return 0. # default exit
# %%
@njit
def aperture_grid(arr, p0=np.array([0.,0.]), sq=np.array([[1.,0.],[0.,1.]]),
                  q0=np.array([0.,0.]), qlim=1., qsmt=0.):
    """

    Calculates transmission values for a round smooth aperture on a 2D grid.
    (numba compiled)

    Parameters:
        arr : np.array[shape=(ny,nx)]
            array receiving aperture transmission values
        p0 : np.array[shape=(2)]
            origin pixel
        sq : np.array[shape=(2,2)]
            physical scale
        q0 : np.array[shape=(2)]
            aperture center position on physical scale
        qlim : float
            aperture radius on physical scale
        qsmt : float
            edge smoothness on physical scale
    """
    nd = arr.shape
    #print("dbg aperture_grid: nd = " % nd)
    for j in range(0, nd[1]):
        for i in range(0, nd[0]):
            dp = np.array([i,j],dtype=sq.dtype) - p0
            q = np.dot(sq,dp)
            arr[j,i] = aperture(q, q0, qlim, qsmt)
    return 0
# %%
@njit
def aperture_a_grid(arr, p0=np.array([0.,0.]),
                    sq=np.array([[1.,0.],[0.,1.]]),
                    q0=np.array([0.,0.]),
                    qlim=1., qsmt=0., qa=0., qp=0.):
    """

    Calculates transmission values for a smooth aperture with 
    2-fold edge distortion on a 2D grid.
    (numba compiled)

    Parameters:
        arr : np.array[shape=(ny,nx)]
            array receiving aperture transmission values
        p0 : np.array[shape=(2)]
            origin pixel
        sq : np.array[shape=(2,2)]
            physical scale
        q0 : np.array[shape=(2)]
            aperture center position on physical scale
        qlim : float
            aperture radius on physical scale
        qsmt : float
            edge smoothness on physical scale
        qa : float
            magnitude of 2-fold distortion on physical scale
        qp : float
            direction of positive axis of 2-fold dist. [-Pi, Pi[
    """
    nd = arr.shape
    for j in range(0, nd[1]):
        for i in range(0, nd[0]):
            dp = np.array([i,j]) - p0
            q = np.dot(sq,dp)
            arr[j,i] = aperture_a(q, q0, qlim, qsmt, qa, qp)
    return 0
# %%
@njit
def aperture_dist3_grid(arr, p0=np.array([0.,0.]),
                        sq=np.array([[1.,0.],[0.,1.]]),
                        q0=np.array([0.,0.]), qlim=1., qsmt=0.,
                        qdist=np.array([])):
    """

    Calculates transmission values for a smooth aperture with edge
    distortions up to 3rd order on a 2D grid.
    (numba compiled)

    Parameters:
        arr : np.array[shape=(ny,nx)]
            array receiving aperture transmission values
        p0 : np.array[shape=(2)]
            origin pixel
        sq : np.array[shape=(2,2)]
            physical scale
        q0 : np.array[shape=(2)]
            aperture center position on physical scale
        qlim : float
            aperture radius on physical scale
        qsmt : float
            edge smoothness on physical scale
        qdist : np.array[shape=(na)]
            list of edge distortion parameters, length = na
            0,1 : two-fold distortion x, y
            2,3 : three-fold distortion x, y
            4,5 : four-fold distortion x, y
    """
    nd = arr.shape
    #print("dbg aperture_dist3_grid: nd = ", nd)
    #print("dbg aperture_dist3_grid: na = ", qdist.size)
    for j in range(0, nd[1]):
        for i in range(0, nd[0]):
            dp = np.array([i,j]) - p0
            q = np.dot(sq,dp)
            arr[j,i] = aperture_dist3(q, q0, qlim, qsmt, qdist)
    return 0
#%%
@njit
def hann(n):
    '''
    Returns a 1d preset with a Hann window.
    '''
    return np.sin(np.pi*np.arange(0,n)/n)**2

