# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:23:00 2020
@author: ju-bar

Binning functions

Note: These are test functions. It turns out that they are too slow
from Python to be of good use. However, they work like a charm and
are fast enough when implemented in C++ language.

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
from numba import jit # include compilation support
from emilys.image.geometry import polygon_area, points_close #, poly_intersect
#
#
def edge_connected(cand1, cand2):
    if (cand1[1] == cand2[1]) and (cand1[3] == cand2[3]): return True
    if (cand1[1] == cand2[2]) and (cand1[3] == cand2[4]): return True
    if (cand1[2] == cand2[1]) and (cand1[4] == cand2[3]): return True
    if (cand1[2] == cand2[2]) and (cand1[4] == cand2[4]): return True
    return False
#
#
class pixelgrid:
    """

    Class pixelgrid stores 2d grid parameters and provides some utility
    geometrical functions.

    Attributes
    ----------
        x0, y0 : float
            grid offset coordinates in a reference frame
        dx, dy : float
            grid spacings in reference frame units
        gamma : float
            angle between grid vectors in radians of the reference frame
        rot : float
            rotation of the first grid vector to the first reference vector
            in radians of the reference frame.
        _matrix : numpy.ndarray, shape(2,2), float
            grid basis matrix
        _inverse : numpy.ndarray, shape(2,2), float
            grid basis inverse matrix

        Note: When changing any attribute manually, better call get_matrix
        before using other utility functions to update _matrix and _inverse

    Members
    -------
        offset : getter and setter for numpy.ndarray, shape (2,), float
            related to the grid offset point (x0, y0)
        matrix : returns the grid matrix
        inverse_matrix: returns the inverse grid matrix
        get_matrix : updates the _matrix values from other parameters
            returns the matrix
        pixel_area : returns the area of one pixel in units
            of the reference frame
        pixel_pos : returns the center position of one pixel identified
            by grid indices
        pixel_fidx : returns fractional grid indices for one position
            given in units of the reference frame
        pixel_corners : returns a list of positions in reference frame
            units that represent the corners of a pixel identified by
            grid indices
        pixel_overlap : returns a list of pixels another grid that
            have intersections with pixels of an input grid. Also returns
            a list with the intersection areas.

    """
    def __init__(self, x0=0., y0=0., dx=1., dy=1., gamma=0.5*np.pi, rot=0.):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.gamma = gamma
        self.rot = rot
    @property
    def offset(self):
        return np.array([self.x0, self.y0])
    @offset.setter
    def offset(self, value):
        self.x0 = value[0]
        self.y0 = value[1]
    def get_matrix(self):
        mr = np.array([[np.cos(self.rot),-np.sin(self.rot)],[np.sin(self.rot),np.cos(self.rot)]])
        md = np.array([[self.dx,0.],[self.dy * np.cos(self.gamma),self.dy * np.sin(self.gamma)]]).T
        self._matrix = np.round(np.dot(mr,md), 15)
        self._inverse = np.linalg.inv(self._matrix)
        return self._matrix
    @property
    def matrix(self):
        if not hasattr(self,'_matrix'): 
            self.get_matrix()
        return self._matrix
    @property
    def inverse_matrix(self):
        if not hasattr(self,'_inverse'): 
            self._inverse = np.linalg.inv(self.matrix)
        return self._inverse
    @property
    def pixel_area(self):
        return np.linalg.det(self.matrix)
    @property
    def pixel_max_size(self):
        d0 = np.dot(self.matrix, [1.,  1.])
        d1 = np.dot(self.matrix, [1., -1.])
        return max(np.sqrt(d0[0]**2 + d0[1]**2), np.sqrt(d1[0]**2 + d1[1]**2))
    @property
    def pixel_min_size(self):
        mt = self.matrix.T
        ap = np.array([-mt[0,1], mt[0,0]])
        bp = np.array([-mt[1,1], mt[1,0]])
        return min(np.abs(np.dot(bp, mt[0])) / np.sqrt(np.dot(mt[0],mt[0])), np.abs(np.dot(ap, mt[1])) / np.sqrt(np.dot(mt[1],mt[1])))
    def pixel_idx(self, ix, iy, ndimx, ndimy, periodic=False):
        jx = ix; jy = iy
        if periodic: # periodic wrap
            jy = (iy % ndimy); jx = (ix % ndimx)
        else:
            if (ix < 0) or (ix >= ndimx) or (iy < 0) or (iy >= ndimy):
                return -1
        return jx + jy * ndimx
    def idx_pixel(self, idx, ndimx):
        ix = idx % ndimx
        iy = int((idx - ix) / ndimx)
        return [ix, iy]
    def pixel_pos(self, ix, iy):
        return self.offset + np.dot(self.matrix, [ix,iy])
    def pos_pixel(self, pos):
        return np.dot(self._inverse,(pos - self.offset))
    def pixel_fidx(self, pos):
        return np.dot(self.inverse_matrix, (pos - self.offset))
    def pos_in_pixel(self, pos, ix, iy):
        fidx = self.pixel_fidx(pos) - np.array([ix,iy], dtype=float)
        if (fidx[0]>=-0.5 and fidx[0]<0.5 and fidx[1]>=-0.5 and fidx[1]<0.5): return True
        return False
    def npos_in_pixel(self, pos, ix, iy):
        fidx = self.pixel_fidx(pos) - np.array([ix,iy], dtype=float)
        if (fidx[0]>=-0.5 and fidx[0]<0.5 and fidx[1]>=-0.5 and fidx[1]<0.5): return 1
        return 0
    def pixel_corners(self, ix, iy, closed=False):
        l_cor = np.array([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]])  + np.array([ix,iy])
        l_ver = np.dot(self.matrix, l_cor.T).T + self.offset
        if closed:
            return np.append(l_ver,l_ver[0]).reshape((len(l_ver)+1,2))
        return l_ver
    def other_pixel_corners_in_pixel(self, pix, other, pix_other):
        m_inside = np.zeros([4,3], dtype=float)
        c_other = other.pixel_corners(pix_other[0], pix_other[1])
        for i in range(0, 4):
            m_inside[i,0] = self.pos_in_pixel(c_other[i],pix[0],pix[1])
            m_inside[i,1:3] = c_other[i]
        return m_inside
    def register_other_edges(self, other, tol=1.E-6):
        m_self = self.matrix.T
        self._v_self = np.array([m_self[0], m_self[1], -m_self[0], -m_self[1]]) # vectors between corners ... this grid
        m_other = other.matrix.T
        self._v_other = np.array([m_other[0], m_other[1], -m_other[0], -m_other[1]]) # vectors between corners ... other grid
        if hasattr(self, '_reg_edge_crossing'): del self._reg_edge_crossing
        self._reg_edge_crossing = np.zeros([4,4], dtype=int)
        if hasattr(self, '_reg_edge_matrix'): del self._reg_edge_matrix
        self._reg_edge_matrix = np.zeros([4,4,2,2], dtype=float)
        for i in range(0,4): # loop over edges of pix in this grid
            for j in range(0,4): # loop over edges of pix_other in other grid
                MDP12 = np.array([self._v_self[i], -self._v_other[j]]).T
                if np.abs(np.linalg.det(MDP12)) > tol: # this means crossing edges
                    self._reg_edge_crossing[i, j] = 1
                    self._reg_edge_matrix[i, j, :, :] = np.linalg.inv(MDP12)
    def other_pixel_to_pixel_edge_crossings(self, pix, other, pix_other, tol=1.E-6):
        if ((not hasattr(self, '_reg_edge_crossing')) or (not hasattr(self, '_reg_edge_matrix'))):
            self.register_other_edges(other, tol)
        c_self = self.pixel_corners(pix[0], pix[1]) # corners of pix in this grid
        c_other = other.pixel_corners(pix_other[0], pix_other[1])
        m_cross = np.zeros([4,4,5], dtype=float) # result matrix of corner x corner x [cross_type, p_x, p_y, t1, t2]
        for i in range(0,4): # loop over edges of pix in this grid
            for j in range(0,4): # loop over edges of pix_other in other grid
                if self._reg_edge_crossing[i, j] > 0: # this means crossing edges
                    DPL = c_other[j] - c_self[i] # distance vector between starting corners
                    t = np.dot(self._reg_edge_matrix[i, j], DPL)
                    if (t[0]>=0.0 and t[0]<=1.0 and t[1]>=0.0 and t[1]<=1.0): # crossing in both edge ranges
                        m_cross[i, j, 0] = 1
                        p = c_self[i] + t[0] * self._v_self[i]
                        m_cross[i, j, 1:3] = p[0:2]
                        m_cross[i, j, 3:5] = t[0:2]
        return m_cross
    def other_pixel_intersect(self, pix, other, pix_other, tol=1.E-6):
        cand = np.zeros([24,5], dtype=int) # candidate vertices index table
        cand_pt = np.zeros([24,2], dtype=float) # candidate positions
        poly = [] # finals intersection polygon
        conn = [0, 0, 0, 0] # flags if edges of the other grid are part of the intersection
        m_cc_1 = self.other_pixel_corners_in_pixel(pix, other, pix_other) # list of other corner pixels with inside this pixel flags
        m_cc_2 = other.other_pixel_corners_in_pixel(pix_other, self, pix) # list of this corner pixels with inside other pixel flags
        m_ce = self.other_pixel_to_pixel_edge_crossings(pix, other, pix_other, tol) # matrix of edge crossing data
        # fill candidate list
        for i in range(0, 4):
            if m_cc_1[i,0] > 0.5: # corners of the other grid pixel that are in this grid pixel -> assigned to grid "1"
                cand[i, :] = [1, 1, 1, i, (i - 1)%4]
                cand_pt[i, 0:2] = m_cc_1[i, 1:3]
            if m_cc_2[i,0] > 0.5: # corners of this grid pixel that are in the other grid pixel -> assigned to grid "0"
                cand[i + 4, :] = [1, 0, 0, i, (i-1)%4]
                cand_pt[i + 4, 0:2] = m_cc_2[i, 1:3]
        for i in range(0, 4):
            for j in range(0, 4):
                if m_ce[i, j, 0] > 0.5: # edge crossing between corners of both pixels
                    idx = 8 + j + i * 4
                    cand[idx,:] = [1, 0, 1, i, j] # on edge index i on this grid pixal and edge index j on the other grid pixel
                    cand_pt[idx, 0:2] = m_ce[i, j, 1:3]
        # run intersection polynomial logics on the candidate list and fill the polynomial
        nc = np.sum(cand[:,0]) # number of candidates
        ic_prev = -1 # index of the previous candidate
        while nc > 0: # still candidates to handle
            for ic in range(0, 24): # loop over candidate array
                ic_add = -1 # new candidate index to add
                if cand[ic,0] > 0: # this is an unhandled candidate
                    if (ic_prev >= 0): # compare candidate ic against the previous for connection
                        if edge_connected(cand[ic], cand[ic_prev]):
                            ic_add = ic
                        # if ((cand[ic,1] == cand[ic_prev,1] and cand[ic,3] == cand[ic_prev,3]) or
                        #     (cand[ic,1] == cand[ic_prev,2] and cand[ic,3] == cand[ic_prev,4]) or
                        #     (cand[ic,2] == cand[ic_prev,1] and cand[ic,4] == cand[ic_prev,3]) or
                        #     (cand[ic,2] == cand[ic_prev,2] and cand[ic,4] == cand[ic_prev,4])):
                        #     ic_add = ic
                    else:
                        ic_add = ic
                if ic_add >= 0: # add the candidate to the intersection polygon
                    poly.append(cand_pt[ic_add])
                    if cand[ic_add,1] == 1: conn[cand[ic_add,3]] = 1
                    if cand[ic_add,2] == 1: conn[cand[ic_add,4]] = 1
                    cand[ic_add, 0] = 0 # no longer a candidate
                    nc -= 1 # decrement candidate count
                    ic_prev = ic_add
                    for jc in range(0, 24): # loop over candidate list and inactivate close points
                        if jc == ic_add: continue # skip current point
                        if 0 == cand[jc, 0]: continue # skip inactive points
                        if points_close(cand_pt[ic_add], cand_pt[jc], tol): # close point -> deactivate
                            cand[jc, 0] = 0
                            nc -= 1 # decrement candidate count
                    break # stop looping ic, new run from while ...
            if ic_add < 0: break # finish as nothing was added
        return (poly, conn)
    def pixel_overlap(self, pix, grid, i_max, r_min=1, periodic=False, debug=False, tol=1.E-6):
        """

        Calculates the overlap of pixel pix of this grid with pixels of the
        input grid assuming that i_max is the maximum index of the input grid.

        Parameters
        ----------
            pix : array of 2 int
                pixel index of this grid
            grid : pixelgrid
                input grid that is compared to this grid
            i_max : array of 2 int
                input grid maximum index (colmns, rows)
            r_min : int, default -1
                minimum search range, values smaller than 0 deactivate search range limit
            periodic : boolean, default: False
                flags use of periodic boundary conditions
            debug : boolean, default: False
                flags debug output via print
            tol : float, default 1.E-6
                tolerated minimum fractional intersection area to consider

        Returns
        -------
            l_pix, l_fac, cover, rng
            l_pix : list of int
                pixel indices when interpreting the input grid as row-wise stream
            l_fac : list of float
                pixel relative intersection factors (normalized to pixel areas)
                these numbers are fractions of input grid pixels covered by the
                pixel pix of this grid
            cover : float
                total relative intersection of pix by the list of pixels l_pix
            rng : int
                index range checked


        """
        fi = grid.pixel_fidx(self.pixel_pos(*pix)) # closest pixel on input grid, fractional index
        ni = np.round(fi,0).astype(int) # closest pixel on input grid, integer index
        nir = 0 # current search radius on grid 1
        nir_min = r_min
        acov = 0.0 # coverage
        dcov = 1.0 # change of coverage
        l_idx = [] # list of grid 1 pixels (result)
        l_fac = [] # list of grid 1 factors (result)
        l_hnd = [] # list of handled pixel indices (internal)
        while (dcov > tol) or (nir < nir_min): # coverage changes
            dcov = 0.
            for i1 in range(ni[1]-nir, ni[1]+nir+1): # loop input grid rows
                if periodic:
                    j1 = i1 % i_max[1] # periodic wrap
                else:
                    if (i1 < 0 or i1 >= i_max[1]):
                        continue; # skip out of bounds
                    j1 = i1
                i2 = j1 * i_max[0]
                for i0 in range(ni[0]-nir, ni[0]+nir+1): # loop input grid columns
                    if periodic:
                        j0 = i0 % i_max[0] # periodic wrap
                    else:
                        if (i0 < 0 or i0 >= i_max[0]):
                            continue # skip out of bounds
                        j0 = i0
                    idx1 = j0 + i2 # grid pixel index
                    if idx1 < 0: continue
                    if idx1 in l_hnd: continue # already handled, skip
                    l_hnd.append(idx1) # mark this pixel as handled
                    l_vf, con = self.other_pixel_intersect(pix, grid, [j0, j1], tol=tol)
                    # l_vf = poly_intersect(grid.pixel_corners(i0,i1), self.pixel_corners(*pix), tol=tol, debug=debug) # get intersection
                    if len(l_vf) > 0: # there is intersection
                        nir_min = -1 # deactivate external search range pushing
                        afac = polygon_area(l_vf) / grid.pixel_area
                        if (afac > tol):
                            acov += afac
                            dcov += afac
                            l_idx.append(idx1)
                            l_fac.append(afac)
                            if debug: print('dbg (pixel_overlap): added pixel [{:d},{:d}] ({:d}) with overlap: {:.6G}'.format(i0, i1, idx1, afac))
            nir +=1
        return (l_idx, l_fac, acov, nir)
#
#
@jit
def hfac_rebin(a, a_hash_idx, a_hash_fac):
    """

    Rebins array a to a new array by a given factor hash lists.
    The length of the output array is given by the length of the hash lists.
    Each item of a_hash_idx is a list of indices identifying items of a
    that contribute to an item of the output. Each item of a_hash_fac is
    a list of factors determining by how much an item of a contributes.

    Parameters
    ----------

        a : numpy.ndarray of shape (n)
            input array
        a_hash_idx : list
            list of lists of indices
        a_hash_fac : list
            list of lists of factors

    Returns
    -------
        numpy.ndarray of shape (len(a_hash_idx))

    Remarks
    -------
        This is meant for flattened input and output arrays.

    """
    n = len(a)
    m = len(a_hash_idx)
    b = np.zeros(m, dtype=a.dtype)
    for j in range(0, m):
        for i in range(0, len(a_hash_idx[j])):
            idx = a_hash_idx[j][i]
            if (idx >= 0 and idx < n): b[j] += (a_hash_fac[j][i] * a[idx])
    return b