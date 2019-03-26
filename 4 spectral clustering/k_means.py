# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:03:25 2018

@author: WednesdayIsAGirl
"""
from math import fabs, sqrt
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class Kmeans(object):
    '''
    x -- input data
    n -- devided into n group
    '''
    def __init__(self, x):
        self.x = x
        self.dimension = x.shape[1]
        self.datasize = x.shape[0]
        self.K = []
    
    # generate initial seed
    def initial(self):
        rerank = np.random.permutation(self.datasize)
        self.K = [self.x[ik, :] for ik in rerank[:self.ngroup]]
        return rerank[:self.ngroup]
        
    # distance between two point, default is linear
    def distance(self, x, y, DTYPE = "linear"):
        # linear distance
        if DTYPE == "linear":
            dxy = y - x
            dis = 0.0
            for di in dxy:
                dis += (di * di)
            return dis
    
    # recalculate the center of group    
    def center(self, K, xindex):
        sumxy = K
        for ix in xindex:
            sumxy = sumxy + self.x[ix]
        centerxy = sumxy / (len(xindex) + 1)
        return centerxy
    
    # k-means
    def k_means(self):
        d_sum = 0.0
        index_list = [ix for ix in range(self.datasize)]
        group_indexs = [[] for ix in range(self.ngroup)]
        centers = deepcopy(self.K)
        while( True ):
            if not index_list:
                break
            ig_min = 0
            index_min = 0
            dis_min = float("inf")
            for ig in range(self.ngroup):
                for ix in index_list:
                    dis = self.distance(centers[ig], self.x[ix])
                    if dis < dis_min:
                        ig_min = ig
                        index_min = ix
                        dis_min = dis
            group_indexs[ig_min].append(index_min)
            index_list.remove(index_min)
            centers[ig] = self.center(centers[ig], group_indexs[ig])
            d_sum += sqrt(dis_min)
        return group_indexs, d_sum
    
    # training, chose the besk key points  
    def train(self, n, epoch = 10):
        self.ngroup = n
        d_sum = np.zeros(epoch)
        Ks = [[] for ix in range(epoch)]
        # generate 10 K-means seed randomly, chose the one who has minnest distance sum
        for i in range(epoch):
            self.initial()
            Ks[i] = deepcopy(self.K)
            group_indexs, d_sum[i] = self.k_means()
        
        # chose K whose distance is the minnest
        index1 = np.where(d_sum == np.min(d_sum))
        self.K = Ks[index1[0][0]]
        group_indexs, d_sum0 = self.k_means()
        
        print("training done")
        #self.paint(group_indexs)

    # paint points
    def paint(self, group_indexs):
        x1 = self.x[group_indexs[0], 0]
        y1 = self.x[group_indexs[0], 1]
        l1, = plt.plot(x1, y1, 'b.')
        x2 = self.x[group_indexs[1], 0]
        y2 = self.x[group_indexs[1], 1]
        l2, = plt.plot(x2, y2, 'r.')
        # paint Key points
        plt.scatter(self.K[0][0], self.K[0][1], marker='p',c='',edgecolors='b', linewidths = 5)
        plt.scatter(self.K[1][0], self.K[1][1], marker='p',c='',edgecolors='r', linewidths = 5)


'''
------------------------------------------------------------------------------
 main program
'''
if __name__ == '__main__':
    # prepare data
    data = []
    group = []
    with open('../data/iris.data') as infile:
        print("open file successfully!\n")
        for line in infile.readlines():
            linea = line.split(",")
            data.append(linea[:2])
    input0 = np.array(data).astype(float)
    
    # rerank data
    rerank = np.random.permutation(input0.shape[0])
    lens = int(input0.shape[0]*0.8)
    input_train  =  input0[rerank[:lens], :]
    input_test   =  input0[rerank[lens+1:],:]
                
    # training
    km1 = Kmeans(input_train)
    km1.train(2)










