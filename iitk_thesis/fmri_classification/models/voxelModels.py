import numpy as np
from sklearn.svm import LinearSVC

class baseline:
    
    def __init__(self, xTrain, yTrain, xTest, yTest):
        self.clf = LinearSVC()
        self.__xTrain = xTrain
        self.__xTest = xTest
        self.__yTrain = yTrain
        self.__yTest = yTest
        self.score = None
    
    def train(self):
        self.clf.fit(self.__xTrain, self.__yTrain)
        self.score = self.clf.score(self.__xTest, self.__yTest)

        
class voxel:
    
    def __init__(self, xTrain, yTrain, xTest, yTest, nVoxels=1000):
        self.clf = LinearSVC()
        self.__xTrain = xTrain
        self.__xTest = xTest
        self.__yTrain = yTrain
        self.__yTest = yTest
        self.nVoxels = nVoxels
        self.score = None
    
    def transform(self, voxelScore):
        sortedIndex = np.argsort(voxelScore)[::-1]
        self.__xTrain = self.__xTrain[:, sortedIndex[:self.nVoxels]]
        self.__xTest = self.__xTest[:, sortedIndex[:self.nVoxels]]
        
    def train(self):
        self.clf.fit(self.__xTrain, self.__yTrain)
        self.score = self.clf.score(self.__xTest, self.__yTest)
