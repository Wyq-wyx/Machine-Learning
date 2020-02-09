#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from numpy import *
import operator

#创建数据集和标签
def createDataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#KNN算法    inX:用于分类的输入向量
def classify0(inX, dataSet, labels, k):
    #距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #把输入向量沿最低维度方向复制datasize倍，沿次低维度复制1倍
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}
    #选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    #排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] #返回类别出现次数最多的分类名称

group,labels = createDataset()
print("result:", classify0([0,0.2],group,labels,3))

#将文本记录转换为训练样本矩阵和类标签向量
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines) #得到文件行数
    retrunMat = zeros((numberOfLines,3)) #创建训练样本矩阵
    classKabekVedtor = [] #创建类标签向量
    index = 0
    for line in arrayOLines:
        line = line.strip() #截取掉所有回车字符
        listFromLine = line.split('\t') #用\t将上一步得到的整行数据分割成一个元素列表
        retrunMat[index,:] = listFromLine[0:3] #取前3个元素存入矩阵中
        classKabekVedtor.append(int(listFromLine[-1])) #负索引-1表示列表中最后一列元素
        index += 1
    return retrunMat,classKabekVedtor

# datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')

#用Matplotilb创建散点图
import matplotlib
import matplotlib.pyplot as plt
def showPlot(datingDataMat, datingLabels):
    type1_x = []; type1_y = []
    type2_x = []; type2_y = []
    type3_x = []; type3_y = []
    for i in range(len(datingLabels)):
        if datingLabels[i] == 1: #label=1
            type1_x.append(datingDataMat[i][1])
            type1_y.append(datingDataMat[i][2])
        if datingLabels[i] == 2: #label=2
            type2_x.append(datingDataMat[i][1])
            type2_y.append(datingDataMat[i][2])
        if datingLabels[i] == 3: #label=3
            type3_x.append(datingDataMat[i][1])
            type3_y.append(datingDataMat[i][2])
    plt.figure()
    plt.subplot(111)
    type1 = plt.scatter(type1_x, type1_y, c = 'red', marker='.')
    type2 = plt.scatter(type2_x, type2_y, c = 'green', marker='.')
    type3 = plt.scatter(type3_x, type3_y, c = 'blue', marker='.')
    plt.xlabel("ice-cream")
    plt.ylabel("video game")
    plt.legend((type1, type2, type3), ("Didn't Like", "Small Doses", "Large Doses"), loc = 0)
    plt.show()

# showPlot()

#特征值归一化  newValue = (oldValue - min)/(max - min)
def autoNorm(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normDataset = dataset - tile(minVals,(m,1))
    normDataset = normDataset / tile(ranges,(m,1))
    return normDataset, ranges, minVals

# normMat, ranges, minVals = autoNorm(datingDataMat)

#测试
def datingClassTest():
    testRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * testRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back result: %d, the real answer is: %d" % (classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print("the totla errror rate is: %f" % (errorCount / float(numTestVecs)))

# datingClassTest()

# 预测
def classifyPerson():
    resultList = ['Didn\'t Like', 'Small Does', 'Large Does']
    percentTats = float(input("percentage of time spent playing video games: "))
    ffMiles = float(input("frequent flier miles earned per year: "))
    iceCream = float(input("liters of ice cream consumed per year: "))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normedMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normedMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult - 1]) # classifyerResut-1:分类结果为123，而resultlist中排序是012

# classifyPerson()
import os
#将32x32的图像转换为1x1024的向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# testvec = img2vector('digits/testDigits/0_13.txt')
# print(testvec[0,0:31])
# print(testvec[0,31:63])

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits') #获取目录
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i] #读取每行数据文件名称
        fileStr = fileNameStr.split('.')[0] #split文件，通过识别”.“,[0]代表除去后面的，即txt
        classNumStr = int(fileStr.split('_')[0]) #split文件，通过识别”_”，[0]除去了0_3后面的序号3，保留0
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('digits/testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # print("the classifier came back result with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is: %f" % (errorCount/float(mTest)))

handwritingClassTest()

