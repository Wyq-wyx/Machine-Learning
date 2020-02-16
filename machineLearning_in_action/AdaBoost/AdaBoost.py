#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from numpy import *

def loadSimpData():
    dataMat = matrix([[1., 2.1], [1.5, 1.6], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat,classLabels

# datMat, classLabel = loadSimpData()

#通过阈值比较对数据进行分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))  # 初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # 如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0  # 如果大于阈值,则赋值为-1
    return retArray


#遍历stumpClassify()函数所有的可能输入值，并找到数据集上的最佳单层决策树
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = float('inf')
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:  # 大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)  # 计算阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 计算分类结果
                errArr = mat(ones((m, 1)))  # 初始化误差矩阵
                errArr[predictedVals == labelMat] = 0  # 分类正确的,赋值为0
                weightedError = D.T * errArr  # 计算误差
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                        i, threshVal, inequal, weightedError))
                if weightedError < minError:  # 找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

# D = mat(ones((5,1))/5)
# bestStump,minError,bestClasEst = buildStump(datMat,classLabel,D)
# print('bestStump: ', bestStump)
# print('minError: ', minError)
# print('bestClasEst: ', bestClasEst)


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 初始化权重
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # 构建单层决策树
        print("D:", D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
                                                                     #max(error, le - 16)用于确保在没有错误时不会发生除零溢出
        bestStump['alpha'] = alpha  # 存储弱学习算法权重
        weakClassArr.append(bestStump)  # 存储单层决策树
        print("classEst: ", classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # 计算e的指数项
        D = multiply(D, exp(expon))
        D = D / D.sum()  # 根据样本权重公式，更新样本权重
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))  # 计算误差
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0: break  # 误差为0，退出循环
    return weakClassArr, aggClassEst

# print(adaBoostTrainDS(datMat, classLabel))

#利用训练出的分类器进行分类
def adaClassify(datatoclass, classifierarr):
        datamatrix = mat(datatoclass)
        m = shape(datamatrix)[0]
        aggclassest=mat(zeros((m,1)))
        for i in range(len(classifierarr)):
            classest = stumpClassify(datamatrix,classifierarr[i]['dim'],classifierarr[i]['thresh'],classifierarr[i]['ineq'])
            aggclassest += classifierarr[i]['alpha']*classest
            print (aggclassest)
        return sign(aggclassest)

# weakClassArr, aggClassEst = adaBoostTrainDS(datMat, classLabel)
# print(adaClassify([[0,0],[5,5]], weakClassArr))


#应用
def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

# dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
# weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
# testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
# predictionsTrain = adaClassify(dataArr, weakClassArr)
# errArrTrain = mat(ones((len(dataArr), 1)))
# predictionsTest = adaClassify(testArr, weakClassArr)
# errArrTest = mat(ones((len(testArr), 1)))
# print('训练集的错误率:%.3f%%' % float(errArrTrain[predictionsTrain != mat(LabelArr).T].sum() / len(dataArr) * 100))
# print('测试集的错误率:%.3f%%' % float(errArrTest[predictionsTest != mat(testLabelArr).T].sum() / len(testArr) * 100))

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)  # 绘制光标的位置
    ySum = 0.0  # 用于计算AUC
    numPosClas = sum(array(classLabels) == 1.0)  # 统计正类的数量
    yStep = 1 / float(numPosClas)  # y轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)  # x轴步长
    sortedIndicies = predStrengths.argsort()  # 预测强度排序
    fig = plt.figure()
    fig.clf()
    plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0;
            delY = yStep
        else:
            delX = xStep;
            delY = 0
            ySum += cur[1]  # 高度累加
        plt.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='r')  # 绘制ROC
        cur = (cur[0] - delX, cur[1] - delY)  # 更新绘制光标的位置
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.axis([0, 1, 0, 1])
    print('the Area Under the Curve is:', ySum * xStep)  # 计算AUC
    plt.show()

dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
plotROC(aggClassEst.T, LabelArr)


