import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

class verbModel:
    def __init__(self, xTrain, yTrain, xTest, yTest):
        self.regr = LinearRegression()
        self.clf = LinearSVC()
        self.__xTrain = xTrain
        self.__xTest = xTest
        self.__yTrain = yTrain
        self.__yTest = yTest
        self.__verbVectorsTrain = None
        self.__verbVectorsTest = None
        self.__layer1 = None
        self.score = None
    
    def layer1(self):
        print('layer 1 training')
        self.regr.fit(self.__xTrain, self.__verbVectorsTrain)
        self.__layer1 = self.regr.predict(self.__xTest)
        print()
    
    def parseVerbVectors(self, nLabels, verbVectors):
        verbVectorsTrain = []
        for yl in self.__yTrain:
            i = nLabels.index(yl)
            verbVectorsTrain.append(verbVectors[i])
        self.__verbVectorsTrain = verbVectorsTrain

        verbVectorsTest = []
        for yl in self.__yTest:
            i = nLabels.index(yl)
            verbVectorsTest.append(verbVectors[i])
        
        self.__verbVectorsTest = verbVectorsTest
    
    def layer2(self, nLabels, verbVectors):
        print('layer 2 training')
        self.clf.fit(verbVectors, nLabels)
        self.score = self.clf.score(self.__layer1, self.__yTest)