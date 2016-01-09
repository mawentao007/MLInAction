#coding:utf-8
__author__ = 'marvin'

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    #print(diffMat)
    sqDiffMat = diffMat ** 2
    #print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    #print(sqDistances)
    distances = sqDistances ** 0.5
    sortedDistances = distances.argsort()     #按照排序后的顺序确定索引,不再是按照所在位置从0开始
    #print(sortedDistances[0])
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1   #统计相应labels的值的数量
    sortedClassCount = sorted(classCount.iteritems(),
        key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix("../machinelearninginaction/Ch02/datingTestSet.txt")       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('../machinelearninginaction/Ch02/digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('../machinelearninginaction/Ch02/digits/trainingDigits/%s'%fileNameStr)
    testFileList = listdir('../machinelearninginaction/Ch02/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('../machinelearninginaction/Ch02/digits/testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("result %d,answer %d\n"%(classifierResult,classNumStr))
        if(classifierResult != classNumStr):errorCount += 1.0

    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))


if __name__ == '__main__':
    tran = {"didntLike":0,"smallDoses":1,"largeDoses":2}
    group,labels = createDataSet()
    print(classify0([0,0],group,labels,3))
    datingDataMat,datingLabels = file2matrix("../machinelearninginaction/Ch02/datingTestSet.txt")
    print(datingDataMat)
    print(datingLabels)
    intDatingLabels = [tran.get(x) for x in datingLabels ]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0 * array(intDatingLabels),15.0*array(intDatingLabels))
    #plt.show()
    normMat,ranges,minVals = autoNorm(datingDataMat)
    print(normMat)
    datingClassTest()
    handwritingClassTest()