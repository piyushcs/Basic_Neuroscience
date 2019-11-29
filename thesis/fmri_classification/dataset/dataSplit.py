import numpy as np

class dataSplit:
    def __init__(self, X, y):
        self.__X = X
        self.__y = y
        self.__dataDict = {}
        
        # convert data into dictionary form
        self.__covertDatatoDict()
    
    def trainTestSplit(itr=0):
        xTrain, yTrain, xTest, yTest = [], [], [], []
        
        for key in self.__dataDict.keys():
            keyData = self.__dataDict[key]

            for i in range(len(keydata)):        
                if i == itr: 
                    xTest.append(keydata[i])
                    yTest.append(key)

                else:
                    xTrain.append(keydata[i])
                    yTrain.append(key)

        return np.array(xTrain), np.array(yTrain), np.array(xTest), np.array(ytest)
                        
    def shuffleSplit():
        print('shuffle')
    
    def __covertDatatoDict():
        dataDict = {}
        for i in range(self.__y.shape[0]):
            if self.__y[i] in dataDict:
                dataDict.append(self.__X[i])
            else:
                dataDict = [self.__X[i]]
        
        self.__dataDict = dataDict
