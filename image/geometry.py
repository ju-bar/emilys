# -*- coding: utf-8 -*-
"""
Created on Tue Jun 01 17:50:00 2021
@author: ju-bar

Geometry functions

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
#import timeit

class vertex_form:
    """
    Class vertex_form is a data container providing utility for the
    formation of polygons by vertex chain logics.
    """
    def __init__(self, ref_idx1=0, pt_idx1=0, side_len1=0.0, ref_idx2=0, pt_idx2=0, side_len2=0.0, pt=np.array([0.,0.])):
        self.ref_idx1 = ref_idx1
        self.pt_idx1 = pt_idx1
        self.side_len1 = side_len1
        self.ref_idx2 = ref_idx2
        self.pt_idx2 = pt_idx2
        self.side_len2 = side_len2
        self.pt = pt
    def is_connected(self, other):
        if ((self.ref_idx1 == other.ref_idx1 and self.pt_idx1 == other.pt_idx1) or
            (self.ref_idx1 == other.ref_idx2 and self.pt_idx1 == other.pt_idx2) or
            (self.ref_idx2 == other.ref_idx1 and self.pt_idx2 == other.pt_idx1) or
            (self.ref_idx2 == other.ref_idx2 and self.pt_idx2 == other.pt_idx2)): return True
        return False

def polygon_area(V):
    """
    Returns the area of a polygon.
    """
    x, y = np.array(V).T
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)

def polygon_center(V):
    """
    Returns the center point of a polygon.
    """
    x, y = np.array(V).T
    return np.array([np.mean(x), np.mean(y)])

def points_dist(P1, P2):
    """
    Returns the distance between points P1 and P2.
    """
    DP = P2 - P1
    return np.sqrt(np.dot(DP,DP))

def points_close(P1, P2, tol=1.E-6):
    """
    Returns True if P1 is close to P2.
    """
    if points_dist(P1, P2) < tol: return True
    return False

def edge_cross(P1_1, P1_2, P2_1, P2_2, tol=1.0E-6):
    """
    
    Calculates the cross-over of two edges [P1_1,P1_2]
    and [P2_1,P2_2].

    """
    DP1 = P1_2 - P1_1
    len_1 = np.sqrt(np.dot(DP1, DP1))
    DP2 = P2_2 - P2_1
    len_2 = np.sqrt(np.dot(DP2, DP2))
    DPL = P2_1 - P1_1
    if np.abs(len_1 * len_2 - np.abs(np.dot(DP1,DP2))) < tol: # parallel lines
        dist_l = np.dot(DPL, DP1) # projection of inter-line points on equal line direction 
        if dist_l < tol: # equal lines
            return [2, np.array([0., 0.])] # return signaling no 
        else: # parallel lines
            return [0, np.array([0., 0.])] # return signaling no 
    # if we get to this line, there should be a single solution to the crossing lines problem
    MDP12 = np.array([DP1,-DP2]).T # np.array([DP1,-DP2]).T
    IDP12 = np.linalg.inv(MDP12)
    t = np.dot(IDP12, DPL)
    return [1, t, P1_1 + t[0] * DP1]

#==============================================================================
# following code of functions is_left and wn_PnPoly refer to ...
# Copyright 2001, softSurfer (www.softsurfer.com)
# This code may be freely used and modified for any purpose
# providing that this copyright notice is included with it.
# SoftSurfer makes no warranty for this code, and cannot be held
# liable for any real or imagined damage resulting from its use.
# Users of this code must verify correctness for their application.

# translated to Python by Maciej Kalisiak <mac@dgp.toronto.edu>

# is_left(): tests if a point is Left|On|Right of an infinite line.
#   Input: three points P0, P1, and P2
#   Return: >0 for P2 left of the line through P0 and P1
#           =0 for P2 on the line
#           <0 for P2 right of the line
#   See: the January 2001 Algorithm "Area of 2D and 3D Triangles and Polygons"
def is_left(P0, P1, P2):
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])
#
# wn_PnPoly(): winding number test for a point in a polygon
#     Input:  P = a point,
#             V[] = vertex points of a polygon
#     Return: wn = the winding number (=0 only if P is outside V[])
def wn_PnPoly(P, V):
    wn = 0   # the winding number counter
    # repeat the first vertex at end
    V = tuple(V[:]) + (V[0],)
    # loop through all edges of the polygon
    for i in range(len(V)-1):     # edge from V[i] to V[i+1]
        if V[i][1] <= P[1]:        # start y <= P[1]
            if V[i+1][1] > P[1]:     # an upward crossing
                if is_left(V[i], V[i+1], P) > 0: # P left of edge
                    wn += 1           # have a valid up intersect
        else:                      # start y > P[1] (no test needed)
            if V[i+1][1] <= P[1]:    # a downward crossing
                if is_left(V[i], V[i+1], P) < 0: # P right of edge
                    wn -= 1           # have a valid down intersect
    return wn
#
#==============================================================================
def poly_intersect(V1, V2, tol=1.0E-6, debug=False):
    """
    
    Calculates a polygon that is the intersection of two polygons.
    Polygons are defined by lists of their vertices V1 and V2.

    Note: This works for convex polygons.

    """
    VP = []
    V3 = []
    n = [len(V1), len(V2)]
    V = [V1, V2]
    # check for vertexes which are contained by one or the other polygon
    #ts = 0.
    #t1 = timeit.default_timer()
    for j in range(0, 2): # loop over polygons
        for i in range(0, n[j]): # loop over 
            wnji = wn_PnPoly(V[j][i], V[1-j]) # vertex j,i in polynom 1-j ?
            if 1 == wnji: # .. yes
                if debug: print('dbg (poly_intersect): vert(',j,',',i,'), P=',V[j][i])
                VP.append(vertex_form(ref_idx1=j, pt_idx1=(i%n[j]), side_len1=0.,
                    ref_idx2=j, pt_idx2=((i-1)%n[j]), side_len2=1.,
                    pt=V[j][i])) # add forming vertex to list
    #t2 = timeit.default_timer()-t1
    #ts += t2
    #print('time (poly_intersect) vertex containing: {:f}'.format(t2))
    # check for crossing of edges between polygons
    #t1 = timeit.default_timer()
    for j in range(0, n[1]): # edges of polygon #1
        pj1 = V2[j]
        pj2 = V2[(j + 1) % n[1]]
        for i in range(0, n[0]): # edges of polygon #2
            pi1 = V1[i]
            pi2 = V1[(i + 1) % n[0]]
            cinfo = edge_cross(pi1, pi2, pj1, pj2, tol)
            if cinfo[0] == 1:
                if (cinfo[1][0] >= 0.0 and cinfo[1][0] <= 1.0 and cinfo[1][1] >= 0.0 and cinfo[1][1] <= 1.0):
                    if debug: print('dbg (poly_intersect): side((0 ,',i,') x (1 ,',j,'): t=', cinfo[1],', P=',cinfo[2])
                    VP.append(vertex_form(ref_idx1=0, pt_idx1=(i%n[0]), side_len1=cinfo[1][0],
                        ref_idx2=1, pt_idx2=(j%n[1]), side_len2=cinfo[1][1],
                        pt=cinfo[2])) # add forming vertex to list
    #t2 = timeit.default_timer()-t1
    #ts += t2
    #print('time (poly_intersect) crossing checks: {:f}'.format(t2))
    nv = len(VP)
    if nv > 0:
        #t1 = timeit.default_timer()
        nh = np.full(nv, 0, dtype=int) # flags handled vertex candidates in VP
        cvf = 0 # start with the first vertex candidate
        firstcycle = True
        while np.sum(nh) < nv: # loop untilt all candidates are handled
            CCV = VP[cvf]
            # find the next candidate that is in some connection to the previous
            if not firstcycle:
                cvf = -1
                for i in range(0, nv):
                    if nh[i] == 1: continue # skip handled candidates
                    if VP[i].is_connected(CCV): cvf = i
                if cvf < 0:
                    break # no connection found, assume finished
            else:
                firstcycle = False # mark that there is no first cycle anymore now
            CCV = VP[cvf]
            # add current candidate as vertex to the output list
            V3.append(CCV.pt)
            # mark all other vertices equal to the current candidate as handled
            for i in range(0, nv):
                if (nh[i]==0):
                    if points_close(CCV.pt, VP[i].pt, tol): # same point, invalidate candidate
                        nh[i] = 1
        #t2 = timeit.default_timer()-t1
        #ts += t2
        #print('time (poly_intersect) intersection generation: {:f}'.format(t2))
    #print('time (poly_intersect) all geometry: {:f}'.format(ts))
    return V3