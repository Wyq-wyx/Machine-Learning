#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from numpy import *

#词表到向量的转换函数
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

def creatVocabList(dataset):
    vocabSet = set([])    #创建一个空集
    for document in dataset:
        vocabSet = vocabSet | set(document)  #创建两个集合的并集
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  #创建一个其中所含元素都为0的向量
    for word in inputSet:   #遍历文档中所有单词
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  #如果出现了词汇表中的单词，则将输出的文档向量中的对应值设置为1
        else:
            print("the word: %s is no in my Vocabulary!" % word)
    return returnVec


# NB分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords) # p0Num = zeros(numWords)  #初始化概率
    p1Num = ones(numWords) # p1Num = zeros(numWords)
    p0Denom = 2.0 # p0Denom = 0.0
    p1Denom = 2.0 # p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]   #向量相加
            p1Denom += sum(trainMatrix[i])  #向量相加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom) # p1Vect = p1Num / p1Denom   #对每个元素做除法
    p0Vect = log(p0Num / p0Denom) # p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive

#NB分类函数
def classifyNB(vec2Classify, p0vec, p1vec, pClass1):
    p1 = sum(vec2Classify * p1vec) + log(pClass1)
    p0 = sum(vec2Classify * p0vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = creatVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,"classified as:", classifyNB(thisDoc,p0v,p1v,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as:", classifyNB(thisDoc, p0v, p1v, pAb))

# testingNB()

#朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec



