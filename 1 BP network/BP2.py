# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:26:12 2018

@author: WednesdayIsAGirl
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
'''
BP network
use sigmord as threshold function
'''
class BPnet(object):
    # initialize，sizes[m] is the number of units of layer m
    # 初始化参数
    def __init__(self, sizes):
        self.sizes = sizes
        self.layers = len(sizes)-2
        # set random weights
        np.random.seed(0)
        self.weights = [np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # units values of each layer
        self.units = [np.zeros(x) for x in sizes]
        self.units_sum = [np.zeros(x) for x in sizes]
    
    # active function
    # 激活函数
    def active_func(self, x, FUNCTYPE):
        if( FUNCTYPE == 'tansig'):
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-1))
        elif( FUNCTYPE == 'sigmoid'):
            return (1 / (1 + np.exp(-x)))
        elif( FUNCTYPE == 'purelin'):
            return x
        else:
            print("active function error")
    
    # degree of active function
    # 激活函数的梯度函数
    def degree_func(self, x, FUNCTYPE):
        if( FUNCTYPE == 'tansig'):
            y = self.active_func(x, 'tansig')
            return 1 - y*y
        elif( FUNCTYPE == 'sigmoid'):
            y = self.active_func(x, 'sigmoid')
            return y * (1-y)
        elif( FUNCTYPE == 'purelin'):
            a = x.shape
            y = np.ones((a[0], a[1]))
            return y
        else:
            print("active function error")
    
    # 前馈函数
    def feedforward(self, x):
        self.units[0] = x.T
        self.units_sum[0] = x.T
        # hidden layer
        for i in range(1, self.layers+1):
            self.units_sum[i] = np.dot(self.weights[i-1], self.units[i-1])
            self.units[i] = self.active_func(self.units_sum[i], 'sigmoid')
        # output layer
        self.units_sum[self.layers+1] = np.dot(self.weights[self.layers], self.units[self.layers])
        self.units[self.layers+1] = self.active_func(self.units_sum[self.layers+1], 'purelin')
        return self.units[self.layers+1]
    
    # 降梯度函数    
    def backprop(self, x, y):
        step = 0.5
        preweights = copy.deepcopy(self.weights)
        lens = x.shape[0]
        
        # output layer
        delta = (y.T - self.units[self.layers+1]) * self.degree_func(self.units_sum[self.layers+1], 'purelin')
        deltaw = np.dot(delta, self.units[self.layers].T)
        self.weights[self.layers] += (step * deltaw / lens)
        
        # hidden layer
        for i in range(self.layers-1, -1, -1):
            weight = preweights[i+1]
            delta = np.transpose(np.dot(delta.T, weight)) * self.degree_func(self.units_sum[i+1], 'sigmoid')
            deltaw = np.dot(delta, self.units[i].T)
            self.weights[i] += (step * deltaw / lens)
      
    # train network, Id - input data, Od - output data
    def train(self, Id, Od):
        train_epoch = 1000
        valid_num = 5
        # validation, devide into 5 group
        lens = Id.shape[0]
        dlens = (int)(lens/valid_num)
        for i in range(valid_num):
            s_index = i*dlens
            if i == 4:
                e_index = lens
            else:
                e_index = (i+1) * dlens
            x = Id[s_index:e_index, :]
            y = Od[s_index:e_index]
            # training
            for i in range(train_epoch):
                self.feedforward(x)
                self.backprop(x, y)
                
    def test(self, Id):
        return self.feedforward(Id)



'''
------------------------------------------------------------------------------
 main program
'''
if __name__ == '__main__':
     # prepare data
    data = []
    group = []
    with open('../data/cloud') as infile:
        print("open file successfully!\n")
        for line in infile.readlines():
            linea = str.split(line)
            data.append(linea[:])
    group = np.zeros(len(data))
    group[0:1024] = 1
    input0 = np.array(data).astype(float)
    output0 = np.array(group).T.astype(float)
    '''
    # prepare data
    data = []
    group = []
    with open('../data/iris.data') as infile:
        print("open file successfully!\n")
        for line in infile.readlines():
            linea = line.split(",")
            data.append(linea[:-1])
            if "setosa" in linea[-1]:
                group.append(1)
            if "versicolor" in linea[-1]:
                group.append(2)
            if "virginica" in linea[-1]:
                group.append(3)
    input0 = np.array(data).astype(float)
    output0 = np.array(group).T.astype(float)
    '''
    # rerank data
    rerank = np.random.permutation(input0.shape[0])
    lens = int(input0.shape[0]*0.8)
    input_train  =  input0[rerank[:lens], :]
    output_train = output0[rerank[:lens]]
    input_test   =  input0[rerank[lens+1:],:]
    output_test  = output0[rerank[lens+1:]]
                
    # training
    sizes = np.array([10, 5, 1])
    nt1 = BPnet(sizes)
    nt1.train(input_train, output_train)
    
    # paint result
    output_esm = nt1.test(input_test)
    data_sum = output_test.shape[0]
    loss = np.zeros(data_sum)
    for i in range(data_sum):
        loss[i] = output_esm[0, i] - output_test[i]
    
    loss_ = np.dot(loss.T, loss) / data_sum
    print("running successfully, the loss is {0:>8.4e}".format(loss_))









