# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 20:31:16 2018

@author: WednesdayIsAGirl
"""
from math import fabs, sqrt, exp
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from k_means import Kmeans

class Spectral(object):
    def __init__(self, x):
        self.datasize = x.shape[0]
        self.dimension = x.shape[1]

    # kernel function
    # 核函数
    def kernel(self, x, y):
        dis = norm(x - y)
        sigma = 1.0
        dis = - dis / (2 * sigma *sigma)
        dis = exp(dis)
        return dis
    
    # generate adjacency matrix
    # 邻接矩阵
    def adjacency_matrix(self, x):
        adj_mat = np.mat(np.zeros((self.datasize, self.datasize)))
        for i in range(self.datasize):
            for j in range(self.datasize):
                adj_mat[i, j] += self.kernel(x[i,:], x[j,:])
        return adj_mat
    
    # generate measure matrix
    # 度矩阵
    def measure_matrix(self, adjacency_matrix):
        ms_mat = np.mat(np.zeros((self.datasize, self.datasize)))
        for i in range(self.datasize):
            ms_mat[i, i] += np.sum(adjacency_matrix[i, :])
        return ms_mat
    
    # N-cut
    # Ncut
    def characteristic_matrix(self, x, cut_type):
        k1 = 2
        charac_mat = np.zeros((self.datasize, k1))
        if cut_type == "Ncut":
            # normalized laplacian matrix
            adj_mat = self.adjacency_matrix(x)
            ms_mat = self.measure_matrix(adj_mat)
            laplace_mat = ms_mat - adj_mat
            ms_m_sqrt = np.sqrt(ms_mat)
            laplace_mat = np.dot(np.dot(ms_m_sqrt.I, laplace_mat), ms_m_sqrt.I)
            lamdas, vectors = np.linalg.eig(laplace_mat)
            for i in range(k1):
                for j in range(self.datasize):
                    charac_mat[j, i] = vectors[j, i+1]
        return charac_mat
        
    # K-means
    # 使用K聚类的方法进行聚类
    def clusting(self, x, charac_m):
        kmeans1 = Kmeans(charac_m)
        kmeans1.train(2, 5)
        group_indexs, d_sum = kmeans1.k_means()
        return group_indexs
    
    # paint 
    def paint(self, x, group_indexs):
        x1 = x[group_indexs[0], 0]
        y1 = x[group_indexs[0], 1]
        l1, = plt.plot(x1, y1, 'r.')
        x2 = x[group_indexs[1], 0]
        y2 = x[group_indexs[1], 1]
        l2, = plt.plot(x2, y2, 'b.')
        plt.legend([l1, l2], \
                   ["group 1 - train", "group 2 - train"])
        
    # training
    def train(self, x, cut_type = "Ncut"):
        charac_mat = self.characteristic_matrix(x, cut_type)
        group_indexs = self.clusting(x, charac_mat)
        self.paint(x, group_indexs)
        return 

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
            if "setosa" in linea[-1]:
                group.append(-1)
            if "versicolor" in linea[-1]:
                group.append(1)
            if "virginica" in linea[-1]:
                group.append(1)
    input0 = np.array(data).astype(float)
    output0 = np.array(group).T.astype(float)
    
    # rerank data
    rerank = np.random.permutation(input0.shape[0])
    lens = int(input0.shape[0]*0.8)
    input_train  =  input0[rerank[:], :]
    input_test   =  input0[rerank[lens+1:],:]
                
    # training
    clust1 = Spectral(input_train)
    clust1.train(input_train)

