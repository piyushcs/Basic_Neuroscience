import numpy as np

class dataSplit:
    def __init__(self, X, y):
        self.__X = X
        self.__y = y
        self.__dataDict = {}
        
        # convert data into dictionary form
        self.__covertDatatoDict()
    
    def trainTestSplit(self, itr=0):
        xTrain, yTrain, xTest, yTest = [], [], [], []

        for key in self.__dataDict.keys():
            keyData = self.__dataDict[key]

            for i in range(len(keyData)):        
                if i == itr: 
                    xTest.append(keyData[i])
                    yTest.append(key)

                else:
                    xTrain.append(keyData[i])
                    yTrain.append(key)

        return np.array(xTrain), np.array(yTrain), np.array(xTest), np.array(yTest)
                        
    def shuffleSplit(self):
        print('shuffle')
    
    def __covertDatatoDict(self):
        dataDict = {}
        for i in range(self.__y.shape[0]):
            if self.__y[i] in dataDict:
                dataDict[self.__y[i]].append(self.__X[i])
            else:
                dataDict[self.__y[i]] = [self.__X[i]]
        
        self.__dataDict = dataDict
