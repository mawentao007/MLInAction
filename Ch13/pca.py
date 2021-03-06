'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    print(dataMat)
    meanRemoved = dataMat - meanVals #remove mean
    #print(meanVals)
    #print(meanRemoved )
    covMat = cov(meanRemoved, rowvar=0)
    #print(covMat )
    eigVals,eigVects = linalg.eig(mat(covMat))
    #print(eigVals)
    #print(eigVects)
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    #print(eigValInd)
    #print(redEigVects)
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    #print(reconMat)
    #print(reconMat)
    return lowDDataMat, reconMat

def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat

if __name__ == "__main__":
    #dataMat = loadDataSet('testSet.txt')
    #lowDMat,reconMat = pca(dataMat,1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=50,c='red')
    # ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=90)
    # plt.show()
    a = [[1,2],[1,2]]
    dataMat = mat(a)
    print dataMat
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #注意一定要加flatten()这个方法，否则矩阵画出的点是错的
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=50,c='red')
    ax.scatter(dataMat[:,0],dataMat[:,1],marker='o',s=90)
    plt.show()


