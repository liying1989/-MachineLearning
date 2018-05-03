# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:31:53 2018

@author: liying30
"""

#入学建模
#目标：
#建立分类器，求解出θ0、θ1、θ2
#设定阈值：根据阈值判断录取结果
#2、要完成的模型
#sigmode：映射到概率的函数
#model：返回预测的结果值
#const：根据参数计算损失值
#gradient：进行参数更新
#currentcy:计算精度
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


data = pd.read_csv('D:/BaiduNetdiskDownload/LogiReg_data.txt',header= None,names = ['exam1','exam2','Admitted'])
#draw
positive = data[data.Admitted == 1]
nagative = data[data.Admitted == 0]

fig,ax =plt.subplots(figsize=(10,5))
ax.scatter(positive.exam1,positive.exam2,s = 30,c='b',marker='o',label ='Admitted')
ax.scatter(nagative.exam1,nagative.exam2,s = 30,c='r',marker='x',label ='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
#drag 
nums = np.arange(-10,10,step =1)
y = sigmoid(nums)
fig,ax =plt.subplots(figsize=(10,5))
ax.plot(nums, sigmoid(nums), 'r')

#function
def sigmoid(x):
    return 1/(1+np.exp(-x))
#建立模型
def Logisic_model(x,theta):
    return(sigmoid(np.dot(x,theta.T)))
#cost function    

#D(hθ(x),y)=−ylog(hθ(x))−(1−y)log(1−hθ(x))
#J(θ)=1n∑i=1nD(hθ(xi),yi)

def cost_funtion(y,x,theta):
    left =  np.multiply(-y,np.log(Logisic_model(x,theta)))
    right = np.multiply(1 - y, np.log(1 - Logisic_model(x, theta)))
    return np.sum(left - right) / (len(x))

theta =  np.zeros([1, 3])
data.insert(0, 'Ones', 1)
orig_data = data.as_matrix() # convert the Pandas representation of the data to an array useful for further computations
cols = orig_data.shape[1]
x = orig_data[:,0:cols-1]
y = orig_data[:,cols-1:cols]


cost_funtion(y,x,theta)
#梯度下降 求出最优的梯度值
#∂J∂θj=−1m∑i=1n(yi−hθ(xi))xij

def gradient(x, y, theta):
    grad = np.zeros(theta.shape)
    error = (Logisic_model(x, theta)- y).ravel()
    for j in range(len(theta.ravel())): #for each parmeter
        term = np.multiply(error, x[:,j])
        grad[0, j] = np.sum(term) / len(x)    
    return grad

STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stopCriterion(type, value, threshold):
    #设定三种不同的停止策略
    if type == STOP_ITER:        return value > threshold
    elif type == STOP_COST:      return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:      return np.linalg.norm(value) < threshold
    

import numpy.random
#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y
import time

def descent(data, theta, batchSize, stopType, thresh, alpha):
    #梯度下降求解
    
    init_time = time.time()
    i = 0 # 迭代次数
    k = 0 # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape) # 计算的梯度
    costs = [cost_funtion(X, y, theta)] # 损失值

    
    while True:
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize #取batch数量个数据
        if k >= n: 
            k = 0 
            X, y = shuffleData(data) #重新洗牌
        theta = theta - alpha*grad # 参数更新
        costs.append(cost_funtion(X, y, theta)) # 计算新的损失
        i += 1 

        if stopType == STOP_ITER:       value = i
        elif stopType == STOP_COST:     value = costs
        elif stopType == STOP_GRAD:     value = grad
        if stopCriterion(stopType, value, thresh): break
    
    return theta, i-1, costs, grad, time.time() - init_time

def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    #import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize==n: strDescType = "Gradient"
    elif batchSize==1:  strDescType = "Stochastic"
    else: strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta


#选择的梯度下降方法是基于所有样本的
n=100
runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)