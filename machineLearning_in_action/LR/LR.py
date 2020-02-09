#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from numpy import *

def loadDataset():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    if inX>=0:
        return 1.0 / (1 + exp(-inX))
    else:
        return exp(inX) / (1 + exp(inX))

#梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001  #目标移动的步长
    maxCycles = 500  #迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# dataArr, labelMat = loadDataset()
# print(gradAscent(dataArr,labelMat))

#分析数据：画出决策边界
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    # weights = wei.getA()  #getA()将weights矩阵转换为数组
    dataMat, labelMat = loadDataset()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    plt.figure()
    plt.subplot(111)
    plt.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    plt.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]
    plt.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# dataArr, labelMat = loadDataset()
# plotBestFit(gradAscent(dataArr,labelMat))

#随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# dataArr, labelMat = loadDataset()
# plotBestFit(stocGradAscent0(array(dataArr),labelMat))

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter = 500):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01  #alpha每次迭代时需要调整
            randIndex = int(random.uniform(0,len(dataIndex)))  #随机选取更新
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(list(dataIndex)[randIndex])
    return weights

# dataArr, labelMat = loadDataset()
# plotBestFit(stocGradAscent1(array(dataArr),labelMat))


'''
------预测病马死亡率
'''
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if classifyVector(array(lineArr),trainWeights) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

multiTest()

'''
对回归系数进行更新的公式为：w：w+alpha*gradient,其中gradient是对参数w求偏导数
推导：https://www.cnblogs.com/zy230530/p/6875145.html
'''
