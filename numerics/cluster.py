# -*- coding: utf-8 -*-
"""
Created on Sun Jul 1 17:15:00 2019
@author: ju-bar

Cluster recognition tools

This code is part of the 'emilys' repository
https://github.com/ju-bar/emilys
published under the GNU General Publishing License, version 3

"""
import numpy as np
# %%
class cluster:
    '''
    
    class cluster

    Handles the assignment of a list item to a cluster, 
    tracking mean and standard deviation of cluster members.

    Members:
        mean : array_like
            mean value of cluster items
        sdev : array_like
            standard deviation of cluster items
        population : array
            list of initial data indices populating the cluster
    
    '''
    def __init__(self, mean, sdev, population):
        self.mean = mean
        self.sdev = sdev
        self.population = population

    def update_stats(self, ldata):
        val0 = ldata[0] * 0.
        self.mean = val0
        self.sdev = val0
        tmpsum = val0
        tmpsqrs = val0
        npop = len(self.population)
        if npop > 0:
            for i in self.population:
                tmpval = ldata[i]
                tmpsum = tmpsum + tmpval
                tmpsqrs = tmpsqrs + tmpval**2
            self.mean = tmpsum / npop
            if npop > 1:
                self.sdev = np.sqrt(np.abs(tmpsqrs / npop - self.mean**2)) \
                            * npop / (npop - 1)

    def add_member_index(self, ldata, index):
        self.population.append(index)
        self.update_stats(ldata)

    def merge(self, other, ldata):
        self.population.extend(other.population)
        self.update_stats(ldata)

class cluster_assignment:
    '''

    Class cluster_assignment

    Handles the cluster assignment and is returned from cluster
    finding algortithms.

    Members:
        lassign : array, int
            forward assignment list (data index -> cluster index)
        lcluster : array, cluster
            list of cluster objects (cluster -> data indices)

    '''
    def __init__(self, nitems):
        '''
        Initialize a cluster_assignment object for a given number of data items.
        '''
        self.lassign = np.full(nitems, -1) # initialize forward list with no assignment
        self.lcluster = [] # initialize empty cluster list

    def add_to_cluster(self, idx_cluster, idx_item, ldata):
        '''
        Adds a data item to an existing cluster.
        '''
        ncl = len(self.lcluster)
        nitems = len(self.lassign)
        if (idx_cluster >= 0 and idx_cluster < ncl and idx_item >= 0 \
            and idx_item < nitems):
            if self.lassign[idx_item] < 0:
                self.lcluster[idx_cluster].add_member_index(ldata, idx_item)
                self.lassign[idx_item] = idx_cluster
                return 0 # success
            return 20 # error, double assignment
        return 10 # error, invalid index

    def add_cluster(self, cluster):
        '''
        Add a cluster, if no conflict is found. Returns list of conflicting items.
        '''
        lconfl = [] # init list of assignment conflicts
        newpopnum = len(cluster.population)
        if newpopnum > 0:
            for i in cluster.population:
                if self.lassign[i] >= 0: # this item already has a cluster
                    lconfl.append(i)
            if len(lconfl) == 0: # no conflict -> add cluster
                self.lcluster.append(cluster)
                icl = len(self.lcluster) - 1
                for i in cluster.population:
                    self.lassign[i] = icl # set forward assignment
                return [] # return empty list of conflicts
            else: # ! conflict, return the list of conflicts
                return lconfl
        return [] # return empty list of conflicts

    def del_cluster(self, idx):
        '''Removes a cluster.'''
        ncl = len(self.lcluster)
        if idx >= 0 and idx < ncl:
            self.lcluster.pop(idx)
            # update the assignment list
            for i in range(0, len(self.lassign)):
                if self.lassign[i] == idx:
                    self.lassign[i] = -1 # erase assignment
                if self.lassign[i] > idx:
                    self.lassign[i] -= 1 # decrement assignment to clusters following the deleted

    def merge_clusters(self, itarget, isource):
        '''
        Merges cluster isource to cluster itarget and updates the assignments.
        
        Returns list of conflicting items.
        '''
        ncl = len(self.lcluster)
        lconfl = []
        if (itarget >=0 and isource >= 0 and itarget < ncl and isource < ncl):
            cl_target = self.lcluster[itarget]
            cl_source = self.lcluster[isource]
            cl_merge = cl_target
            cl_merge.merge(cl_source)
            self.del_cluster(itarget)
            self.del_cluster(isource)
            lconfl = self.add_cluster(cl_merge)
            if len(lconfl) > 0: # This should never happen
                # try reversing the deletion
                self.add_cluster(cl_target)
                self.add_cluster(cl_source)
        return lconfl
# %%
def cluster_l2(ldata, l2_thresh=1.):
    '''

    Creates a list of clusters found in ldata with the l2-norm
    of cluster members better than l2_thresh with respect
    to the cluster mean.

    Parameters:
        ldata : numpy.ndarray, shape(N,2)
            list of 2d point coordinates
        l2_thresh: float
            clustering threshold, must be > 0!

    Return:

        clusters are returned as list of values
            [mean,sdev,population]
            mean : array_like as items of ldata
                mean value of the cluster
            sdev : array_like as items of ldata
                standard deviation of the cluster
            population : array
                list of indices referencing to items of ldata in the cluster
        
        An empty list is returned in case of invalid parameters.

    '''
    nd = ldata.shape
    if len(nd) != 2:
        return []
    ndat = nd[0]
    ndim = nd[1]
    lthr = np.abs(l2_thresh)
    if lthr == 0.:
        return []
    lass = cluster_assignment(ndat)
    nass = 0 # initialize number of assigned values
    for i in range(0, ndat): # loop over all items
        iclclose = -1 # remember index of closest cluster
        dclclose = 1000. * lthr # initialize closest cluster distance to very large
        if len(lass.lcluster) > 0: # try to find possible clusters
            for j in range(0, len(lass.lcluster)): # loop over current clusters
                vdist = lass.lcluster[j].mean - ldata[i] # cluster mean distance vector to query point
                dist = np.sqrt(np.dot(vdist,vdist)) # cluster mean distance
                if dist < l2_thresh: # cluster mean distance is below threshold
                    if dist < dclclose: # closest cluster ?
                        dclclose = dist # remember
                        iclclose = j
        if iclclose < 0: # no good cluster found
            # create and add a new cluster
            cl_new = cluster(ldata[i],np.zeros(ndim),[i])
            if len(lass.add_cluster(cl_new)) == 0: 
                nass += 1 # increase number of assigned items
        else: # cluster found
            if 0 == lass.add_to_cluster(iclclose, i, ldata): # populate the cluster
                nass += 1 # increase number of assigned items
    print('- ',nass,'of',ndat,' items assigned to ', \
          len(lass.lcluster),' clusters')
    return lass
