# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 21:44:20 2018

@author: WednesdayIsAGirl
"""

from math import fabs
import numpy as np
import matplotlib.pyplot as plt

class SVM(object):
    '''
    x -- input data
    y -- group of x
    '''
    def __init__(self, x, y):
        self.C = 10  # punish parameter
        
        self.x = x
        self.y = y
        self.lamda = np.zeros(x.shape[0])
        #self.w = np.zeros((1, x.shape[1]))
        self.K = np.zeros((x.shape[0], x.shape[0]))   # kernel(x, x)
        self.b = 0.0
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                self.K[i, j] = self.kernel(x[i, :], x[j, :])
    
    # kernel function, default is linear
    # 核函数，默认为线性函数
    def kernel(self, x, y, KTYPE = "linear"):
        if KTYPE == "linear":
            return np.dot(x, y.T)
    
    # check if lamda subject to KKT condition
    # 检查拉格朗日因子是否符合KKT条件
    def KKT_check(self, lamda, index):
        outline = 0.05
        #valuei = self.y[index] * (np.dot(self.w, self.x[index].T) + self.b)
        valuei = self.y[index] * (np.dot(np.multiply(self.lamda, self.y), self.K[index, :]) + self.b)
        # lamda = 0 & y(wx+b)>=1
        if (fabs(lamda) < 1e-6 and valuei < 1):
            return False
        # lamda = C & y(wx+b)<=1
        if (fabs(lamda - self.C) < 1e-6 and valuei > 1):
            return False
        if (lamda > 0 and lamda < self.C and fabs(valuei - 1) > outline):
            return False
        return True
    
    # heuristic_method,return index of lamda1 & lamda2
    # 启发式方法，确定需要更新的拉格朗日因子
    def heuristic_method(self):
        for i in range(self.lamda.shape[0]):
            if(self.KKT_check(self.lamda[i], i) == False):
                Error = np.dot(np.multiply(self.lamda, self.y), self.K) - self.b -self.y
                if(Error[i] > 0):
                    indexs2 = np.where(Error == np.min(Error))
                else:
                    indexs2 = np.where(Error == np.max(Error))
                index2 = indexs2[0][0]
                
                '''
                index2 = np.random.randint(self.x.shape[0])
                if index2 == i:
                    index2 = (index2+1) % (self.x.shape[0])
                '''
                return [i, index2]
        # all lamda subject to KKT condition, return [-1, -1]
        return [-1, -1]
    
    # SMO 
    # SMO方法
    def SMO(self, index1, index2):
        # lamda new, unclipped
        y1 = self.y[index1]
        y2 = self.y[index2]
        E1 = np.dot(np.multiply(self.lamda, self.y), self.K[index1, :]) + self.b - y1
        E2 = np.dot(np.multiply(self.lamda, self.y), self.K[index2, :]) + self.b - y2
        #E1 = (np.dot(self.w, self.x[index1].T) + self.b) - y1
        #E2 = (np.dot(self.w, self.x[index2].T) + self.b) - y2
        K11 = y1*y1*self.K[index1, index1]
        K22 = y2*y2*self.K[index2, index2]
        K12 = y1*y2*self.K[index1, index2]
        E12 = E1 - E2
        if(E12 == 0):
            return
        dlamda = y2*(E12) / (K11 + K22 - 2*K12)
        lamda2_new = self.lamda[index2] + dlamda
        # L & H limit
        lamda1 = self.lamda[index1]
        lamda2 = self.lamda[index2]
        if y1 == y2:
            L = max(0, lamda1 + lamda2 - self.C)
            H = min(self.C, lamda1 + lamda2)
        if y1 != y2:
            L = max(0, lamda2 - lamda1)
            H = min(self.C, self.C + lamda2 - lamda1)
        # get lamda2
        if lamda2_new > H:
            lamda2_new = H
        elif lamda2_new < L:
            lamda2_new = L
        else:
            lamda2_new = lamda2_new
        # get lamda1
        lamda1_new = lamda1 + y1*y2*(lamda2 - lamda2_new)
        # update lamda
        self.lamda[index1] = lamda1_new
        self.lamda[index2] = lamda2_new
        
        
        # sometimes lamda is not updated, we rechose a lamda2 and redo SMO 
        if(L == 0 and H == 0) or dlamda == 0:
            return
            
        
        # update b, w is updated  implicitly
        #linear SVM update w
        #self.w = self.w + y1*(lamda1_new - lamda1)*x1 + y2*(lamda2_new - lamda2)*x2
        # update b
        b1 = - E1 - y1*(lamda1_new - lamda1)*self.K[index1, index1] \
        - y2*(lamda2_new - lamda2)*self.K[index2, index1] + self.b    
        b2 = - E2 - y1*(lamda1_new - lamda1)*self.K[index1, index2] \
        - y2*(lamda2_new - lamda2)*self.K[index2, index2] + self.b
        
        if(lamda1_new>0 and lamda1_new<self.C \
           and lamda2_new>0 and lamda2_new<self.C):
            self.b = (b1 + b2) / 2.0
        if(lamda1_new > 0 and lamda1_new < self.C):
            self.b = b1
        elif(lamda2_new > 0 and lamda2_new < self.C):
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0

    # paint
    # 绘制结果
    def paint(self):
        # update w
        self.w = np.dot(np.multiply(self.lamda, self.y), self.x)
        # paint input data
        g1 = self.x[self.y == 1, :]
        g2 = self.x[self.y == -1, :]
        l1, = plt.plot(g1[:,0], g1[:,1], 'b.')
        l2, = plt.plot(g2[:,0], g2[:,1], 'r.')  
        # paint wx + b
        y1 = min(self.x[:,1])
        y2 = max(self.x[:,1])
        x1 = -(self.b + self.w[1]*y1) / self.w[0]
        x2 = -(self.b + self.w[1]*y2) / self.w[0]
        l0, = plt.plot([x1, x2], [y1, y2])
        # paint support vector
        sv = []
        for ilamda in range(self.lamda.shape[0]):
            if fabs(self.lamda[ilamda]) > 1e-5:
               sv.append(self.x[ilamda, :])
        svp = np.array(sv)
        lsv = plt.scatter(svp[:,0], svp[:,1], marker = 'o', c='', edgecolors='k')
        # legend
        plt.legend([l1, l2,  lsv], \
                   ["group 1 - train", "group 2 - train", "support vector"])
        
    # train svm
    # 训练
    def train(self, KTYPE = "linear"):
        epoch = 0
        while(True):
            indexs = self.heuristic_method()
            if indexs[0] == -1:
                break
            self.SMO(indexs[0], indexs[1])
            epoch += 1
            '''
            while True:
                self.SMO(indexs[0], indexs[1])
                epoch += 1
                if(self.KKT_check(self.lamda[indexs[1]], indexs[1])):
                    break
                if epoch > 1000:
                    break
            '''
            if epoch > 10000:
                break
        self.paint()    
        print("training done")
        
    

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
    input_train  =  input0[rerank[:], :]
    output_train = output0[rerank[:]]
                
    # training
    svm1 = SVM(input_train, output_train)
    svm1.train()




