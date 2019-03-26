# ML-practice

##1 实现基础的BP network并在UCI数据集上测试

#程序说明：

可以任意定义神经网络的层数以及每层的节点数目。

数据集进行训练时，会打乱数据集的分类并分为10个子数据集分批BDG，以较少局部最优解的可能性；

隐藏层的激活函数为sigmord或tansig函数，程序默认为sigmord函数；

输出层的激活函数为purelin函数，即线性函数y = x，以扩大输出结果的范围。

#测试结果：

采用UCI的Iris数据集进行分类测试，三种种类的desired output分别定义为1，2，3，利用BP神经网络进行分类。

测试的正确率达到100%，损失函数的值为1.9168e-2。

---
##2 实现线性SVM’s并测试

#程序说明：

SVM实质是解KKT条件下的二次规划问题。

用启发式方法每次寻找最违反KKT条件的两个点，更新相应的两个拉格朗日因子，利用SMO算法更新。每次循环。因为核函数的引入，w可以不显式更新，b需要每次循环更新。

循环直到满足终止条件。

#测试结果：

采用Iris数据集的前两列数据进行测试，前两种类别的desired output定义为1，第三种类别定义为-1。用SVM程序进行分类。

在用python实现的过程中，增加了惩罚因子以实现软分割, 分类结果如下：

![svm1](https://github.com/HelloGoodDay/ML-practice/blob/master/2%20linear%20SVM/4.png)

MALTAB分类的结果如下：

![SVM](https://github.com/HelloGoodDay/ML-practice/blob/master/2%20linear%20SVM/training.png)

可以看出，python实现的SVM在支持向量的选择上并不是特别合适，会出现选择的支持向量并不是最靠近分界线的点的情况，算法待改进

---
##3 实现K-means算法

#程序说明：

K聚类的起始seed是随机选择的，当起始的聚类点选择不合适时，会出现错误的聚类效果。

所以在程序中，随机进行10次K聚类，并选择其中总距离和最小的一次作为输出结果。分类的结果得到明显改善。

#测试结果：

分别使用Iris数据集中的前两列进行测试，以及第2.4列进行测试。

在随机进行K聚类时，结果往往不如人意（如左图）。但对程序进行修改，多次聚类并选择距离和最小的作为输出结果时，可以得到较好的分类结果（如右图）。
 
![K1](https://github.com/HelloGoodDay/ML-practice/blob/master/3%20K-means/kmeans-1-r.png)
![k2](https://github.com/HelloGoodDay/ML-practice/blob/master/3%20K-means/kmeans-10-r.png)
 
但这种分类结果仍然是不能让人满意的，因为分类结果只识别出了点的距离远近，并不能根据点的group 结构来进行聚类，这种缺点可以在谱聚类中得到改善。
当需要聚类的两个数据集间隔比较大，而类间联系又比较紧密时，可以得到很好的聚类结果。

---
##4 实现spectral clustering

#程序说明：

计算邻接矩阵时，核函数采用高斯核函数。

聚类方法采用K聚类。

#测试结果：

![SP1](https://github.com/HelloGoodDay/ML-practice/blob/master/4%20spectral%20clustering/pic2.png)

采用Iris数据集中的前两列作为测试数据进行聚类，相对于K聚类可以得到很好的聚类结果。
 
---
##5 实现dominant-set clustering

#程序说明：

dominant-set聚类的思想是通过不断改变阈值alpha对数据集进行分割，直到分割得到的子数据集都是dominant-set集或只含有一个元素为止。可以实现自动化决定类的数目并进行聚类。这种特点使得它非常适合做图像分割。

但当数据集中点的数目增加时，判断子集是否为dominant-set集的计算量以接近N^N的速度增加(见论文）。

在程序中，为了简化运算，只简单地把数据集分成两个子集。

#测试结果：

为了体现dominant-set聚类的特点，这里没有采用UCI的数据集，而是采用了自己生成的数据集，聚类结果如下，效果很理想。
 
![DS1](https://github.com/HelloGoodDay/ML-practice/blob/master/5%20dominate%20set/pic2.png)
 
实际上，对于Iris数据集这种类内部联系不够明显的数据（K聚类和谱聚类中所用到的），dominant-set分类的效果并不是很好。

![DS2](https://github.com/HelloGoodDay/ML-practice/blob/master/5%20dominate%20set/piic1.png)
 
