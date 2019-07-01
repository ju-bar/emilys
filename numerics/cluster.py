# -*- coding: utf-8 -*-
"""
Created on Sun Jul 1 17:15:00 2019
@author: ju-bar
"""
import numpy as np
# %%
class cluster:
    def __init__(self, mean, sdev, population):
        self.mean = mean
        self.sdev = sdev
        self.population = population

    def update_stats(self, ldata):
        self.mean = 0 * ldata[0]
        self.sdev = 0 * ldata[0]
        tmpval = self.mean
        tmpsum = self.mean
        tmpsqrs = tmpsum
        np = len(self.population)
        if np > 0:
            for i in self.population:
                tmpval = ldata[self.population[i]]
                tmpsum += tmpval
                tmpsqrs += (tmpval * tmpval)
            self.mean = tmpsum / np
            if np > 1:
                self.sdev = np.sqrt( tmpsqrs / np - self.mean * self.mean ) * np / (np - 1)

    def add_member_index(self, ldata, index):
        self.population.append(index)
        self.update_stats(ldata)

    def merge(self, other, ldata):
        self.population.extend(other.population)
        self.update_stats(ldata)

# %%
def cluster_l2(ldata, l2_thresh=1., err=0):
    '''

    Returns a list of clusters found in ldata with the l2-norm
    of cluster members better than l2_thresh with respect
    to the cluster mean

    clusters are returned as list of values
        [mean,sdev,population]
        mean : array_like as items of ldata
            mean value of the cluster
        sdev : array_like as items of ldata
            standard deviation of the cluster
        population : array
            list of indices referencing to items of ldata in the cluster

    err = error code

    '''
    nd = ldata.shape
    if len(nd) != 2:
        err = 10
        return
    ndat = nd[0]
    ndim = nd[1]
    lthr = np.abs(l2_thresh)
    if lthr == 0.:
        err = 11
        return
    lass = np.full(ndat,-1) # initialize assignement table
    nass = 0 # initialize number of assigned values
    lcl = np.array([]) # initialize empty cluster table
    ncl = 0 # initialize number of clusters
    while nass < ndat: # loop to assign all values to clusters
        for i in range(0, ndat): # loop over all items
            if lass[i] >= 0: # unassigned item
                lassi = [] # initialize assignment of this item
                if ncl > 0: # try to find possible clusters
                    for j in range(0, ncl):
                        dist = lcl[j].mean - ldata[i] # cluster mean distance to query point
                        if np.sqrt(np.dot(dist,dist)) < l2_thresh: # cluster mean distance is below threshold
                            lassi.append(j)
                if len(lassi) == 0: # no good cluster found
                    lcl.append([ldata[i],np.zeros(ndim),[i]]) # add new cluster
                    ncl += 1 # raise counts of clusters
                    lass[i] = ncl - 1 # mark this item as assigned
                    nass -= 1 # decrease number of assigned items
                elif len(lassi) == 1: # one good cluster found
                    j = lassi[0] # cluster index in lcl
                    lcl[j].add_member_index(ldata, i) # populate the cluster
                    lass[i] = j # mark this item as assigned
                    nass -= 1 # decrease number of assigned items
                else: # multiple clusters are in threshold range
                    # add new member to the first cluster
                    j = lassi[0] # cluster index in lcl
                    lcl[j].add_member_index(ldata, i) # populate the cluster
                    lass[i] = j # mark this item as assigned
                    nass -= 1 # decrease number of assigned items
                    # re-think the cluster connection, this might not be wise here
                    # could be done at the end
                    for l in range(1,len(lassi)): # connect also the other clusters
                        k = lassi[l]
                        lcl[j].merge(lcl[l],ldata) # merges the populations
                    # remove connected clusters
                    for l in range(len(lassi)-2,-1,-1): # loop through reversed list and exclude the last item (which is the merged cluster)
                        k = lassi[l] # cluster index to remove
                        lcl.pop(k) # remove cluster at index k
                        ncl -= 1 # decrease number of clusters
    return lcl
