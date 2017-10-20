from math import log
import numpy as np
import operator
import csv
import sys
import types
import copy
#sys.stdout.write("def")
sys.argv[1]

def createDataSet( filName):  
    csv_file = csv.reader(open(filName,'r'))
    data = [row for row in csv_file] 
    features = data[0]
    features.remove(features[-1])
    data.remove(data[0])
    dataSet = []
    for i in data:
        i = map(int,i)
        dataSet.append(i) 
    return dataSet,features

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    column = [row[-1] for row in dataSet]
    y = column.count(1)
    n = column.count(0)
    probn = float(n)/numEntries
    proby = float(y)/numEntries
    if n * y !=0: 
        shannonEnt = -probn * log(probn,2) -proby * log(proby,2) #log base 2
    elif n==0 and y!=0:
        shannonEnt = -proby * log(proby,2)
    elif n!=0 and y== 0:
        shannonEnt = -probn * log(probn,2)
    else:
        shannonEnt = 0
    return shannonEnt

def Variance(dataSet):
    K = len(dataSet)
    column = [row[-1] for row in dataSet]
    k1 = column.count(1)
    k0 = column.count(0)
    VI = float(k1*k0)/(K*K)
    return VI

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = np.delete(featVec, axis, 0)
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet,flag):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    if flag == 1:
        baseEntropy = calcShannonEnt(dataSet)
    else:
        baseEntropy = Variance(dataSet)

    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)

        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            if flag ==1:
                newEntropy += prob * calcShannonEnt(subDataSet)
            else:
                newEntropy += prob * Variance(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def createTree(dataSet,labels,flag):
    column = [row[-1] for row in dataSet]
    if column.count(1) == len(column) or column.count(0) == len(column): 
        return column[0]
    if len(dataSet[0]) == 1:
        if column.count(1) > column.count(0):
            return y
        else:
            return n
    if flag == 1:
        bestFeat = chooseBestFeatureToSplit(dataSet,1)
    else:
        bestFeat = chooseBestFeatureToSplit(dataSet,2)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    labels = np.delete(labels, bestFeat, 0)
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    subLabels = []
    for value in uniqueVals:
        for i in range(len(labels)):
            subLabels.append(labels[i])
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        myTree[bestFeatLabel][value] = createTree(subDataSet,subLabels,flag)
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


def myPrint(Tree, key, value,deep):
    if value == -1:
        for subkey in Tree.keys():
            key = subkey
            break
        myPrint(Tree[key],key, 0, deep)
        myPrint(Tree[key],key, 1, deep)
    else:
        
        if type(Tree[value]) == type({}):
            for i in range(deep):
                sys.stdout.write("|")
            sys.stdout.write(str(key)+"=" +str(value)+":")
            print ""
            myPrint(Tree[value],key, -1,deep+1)
        else:
            for i in range(deep):
                sys.stdout.write("|")
            sys.stdout.write(str(key)+"=" +str(value)+":")
            sys.stdout.write(str(Tree[value]))
            print ""

def fff(Tree,key,value,fang,features):
    marker = -1
    if value == -1:
        for subkey in Tree.keys():
            key = subkey
            break
        for i in range(len(features)):
            if key == features[i]:
                marker = i
                break
        if fang[marker] == 1:
            return fff(Tree[key],key, 1 ,fang,features)
        else:
            return fff(Tree[key],key, 0,fang,features)
    else:
        if type(Tree[value]) == type({}):
            return fff(Tree[value],key, -1,fang,features)
        else:
            return Tree[value]

def calc_accuracy(tree, dataset,features):
    right = 0
    for fang in dataset:
        r = fff(tree,0, -1,fang,features)#leaf value
        if r == fang[-1]:
            right = right + 1
    return float(right)/len(dataset)


def P(Tree, key, value,otree,dataset, features):
    if type(Tree) == type(0):
        return
    if value == -1:
        for subkey in Tree.keys():
            key = subkey
            break
        P(Tree[key],key, 0, otree, dataset, features)
        P(Tree[key],key, 1, otree, dataset, features)
    else:
        acc =  calc_accuracy(otree, dataset,features)       
        if type(Tree[value]) == type({}):
            tempvalue = Tree[value]
            Tree[value] = 0
            a = calc_accuracy(otree, dataset,features)
            Tree[value] = 1
            b = calc_accuracy(otree, dataset,features)
            if a > b:
                Tree[value] = 0
                b = a
            if b <= acc:
                Tree[value] = tempvalue           
            P(Tree[value],key, -1,otree, dataset, features)
        else:
            return           

def printResult(Tree):
    if sys.argv[4] == "yes":
        print("======Tree without pruning:======")
        myPrint (tree, 0, -1, 0)

    print("training set accuracy:")
    print calc_accuracy(tree, dataset,features)
    print("validation set accuracy:")
    print calc_accuracy(tree, dataset_validation,features)
    print("test set accuracy:")    
    print calc_accuracy(tree, dataset_test,features)
    prun_tree = {}
    prun_tree = copy.deepcopy(tree)
    if sys.argv[5] == "yes":
        P(prun_tree, 0, -1,prun_tree,dataset_validation, features)
        if sys.argv[4] == "yes":
            print("======Tree with pruning:======")
            myPrint (prun_tree, 0, -1, 0)
        print("=================")
        print("training set accuracy for pruned tree:")
        print  calc_accuracy(prun_tree, dataset,features)
        print("validation set accuracy for pruned tree:")
        print  calc_accuracy(tree, dataset_validation,features) 
        print("test set accuracy for pruned tree:")
        print  calc_accuracy(prun_tree, dataset_test,features)

#===============================
if __name__ == '__main__':  
    dataset,features = createDataSet(sys.argv[1])
    dataset_validation,features_validation =  createDataSet(sys.argv[2])
    dataset_test,features_test =  createDataSet(sys.argv[3])

    tree = createTree(dataset,features,1)
    print('======USEING H1======')
    printResult(tree)
    tree = {}
    tree = createTree(dataset,features,2)
    print('======USEING H2======')
    printResult(tree)




