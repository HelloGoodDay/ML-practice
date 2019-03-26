# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:18:38 2018

@author: WednesdayIsAGirl
"""

from math import fabs, sqrt, exp
from copy import deepcopy
from numpy.linalg import norm, det
from sklearn.datasets import make_blobs, make_circles
import numpy as np
import matplotlib.pyplot as plt


class DominantSet(object):
    def __init__(self):
        return
    
    # kernel function
    # 核函数
    def kernel(self, x, y):
        dis = norm(x - y)
        sigma = 1.0
        dis = - dis / (2 * sigma *sigma)
        dis = exp(dis)
        return dis
    
    # generate adjacency matrix of x 
    # 生成邻接矩阵
    def adjacency_matrix(self, x):
        datasize = x.shape[0]
        adj_mat = np.mat(np.zeros((datasize, datasize)))
        for i in range(datasize):
            for j in range(datasize):
                adj_mat[i, j] += self.kernel(x[i,:], x[j,:])
        return adj_mat
    
    # wi, i belong to S
    # 计算数据集S中元素i的权重
    def weight_i(self, set_list0, indexi):
        set_list = deepcopy(set_list0)
        if (len(set_list) == 1):
            return 1
        else:
            W_i = 0.0
            set_list.remove(indexi)
            for ji in set_list:
                W_i += (self.adj_mat[indexi, ji] - self.adj_mat[set_list[0], ji]) * self.weight_i(set_list, ji)
            return W_i
    
    # same as weight_i
    def weights_i(self, set_list0, indexi):
        set_list = deepcopy(set_list0)
        if (len(set_list) == 1):
            return 1
        else:
            W_i = 0.0
            set_list.remove(indexi)
            for ji in set_list:
                W_i += self.phi_(set_list, ji, indexi) * self.weights_i(set_list, ji)
            return W_i
               
    def phi_(self, set_list0, indexi, indexj):
        ave_w = 0.0
        set_list = deepcopy(set_list0)
        if(len(set_list) == 1):
            ave_w = 0.0
        elif(len(set_list) == 2):
            set_list.remove(indexi)
            ave_w += self.adj_mat[indexi, set_list[0]]
        else:
            set_list.remove(indexi)
            for i in set_list:
                ave_w += self.adj_mat[indexi, i]
        ave_w = ave_w / (len(set_list) + 1)
        phi = self.adj_mat[indexi, indexj] - ave_w
        return phi
    
    # split A into V1, V2
    # 对邻接矩阵进行分割
    def split(self, A, alpha, eps = 1e-4):
        x = np.mat(np.ones(A.shape[0]) / float(A.shape[0])).T
        dx = 1
        while dx > eps:
            x_old = deepcopy(x)
            x = np.multiply(x, np.dot(A - alpha, x))
            x = x / np.sum(x)
            dx = norm(x - x_old)
        vectors = deepcopy(x)
        x = np.array(x)
        cutoff = np.median(x[x > 0])
        index_list1 = np.where(vectors > cutoff)
        index_list2 = np.where(vectors <= cutoff)
        list1 = [indexi for indexi in index_list1[0]]
        list2 = [indexi for indexi in index_list2[0]]
        return list1, list2
    
    # decrease alpha to 0, unitl V is a dominant set
    # 改变alpha的值，直到得到需要的分类结果
    def dominant_set(self, set_list, A):
        alpha = 4
        while True:
            if alpha < 0:
                list1, list2 = self.split(A, 0)
                return list1, list2
            list1, list2 = self.split(A, alpha)
            dom_flag = True
            '''
            for indexi in list1:
                if self.weights_i(list1, indexi) <= 0:
                    dom_flag = False
                break
            '''
            if dom_flag == True:
                return list1, list2
            else:
                alpha = alpha -1      
    
    # paint 
    def paint(self, x, list1, list2):
        plt.plot(x[list1, 0], x[list1, 1], 'bo')
        plt.plot(x[list2, 0], x[list2, 1], 'ro')
    
    # training    
    def train(self, x):
        np.random.seed(1)
        n = 500
        d = 2
        x, y = make_circles(n,shuffle=True,noise=0.03, random_state=2,factor=0.5)
        
        self.adj_mat = self.adjacency_matrix(x)
        set_list = range(x.shape[0])
        list1, list2 = self.dominant_set(set_list, self.adj_mat)
        self.paint(x, list1, list2)

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
    dom1 = DominantSet()
    dom1.train(input_train)