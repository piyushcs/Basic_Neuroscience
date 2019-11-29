from fmriDataset import fmriDataset
import numpy as np
import itertools

class voxelCorr:
    def __init__(self, data, nEpochs):
        self.fmriData = data
        self.nEpochs = nEpochs
        self.__voxelsMatrix = None
        self.__voxelsScore = None
        self.__pairs = None
        
        # setting up the voxel matrix
        self.__setVoxelsMatrix()
        
        # find possible pairs for epochs
        self.__findPossiblePairs()
        
        # calculating voxel correlation score
        self.setVoxelsScore()
    
    ### reformat training data to do voxel correlation analysis
    def __setVoxelsMatrix(self):
        voxelsMatrix = list()
        nVoxels = self.fmriData.shape[1]
        nLabels = (self.fmriData.shape[0])/self.nEpochs
        for n in range(nVoxels):
            voxelFlat = self.fmriData[:, n]
            
            # single voxel rearrangement
            voxelMatrix = list()
            for l in range(int(nLabels)):
                index = l * self.nEpochs    
                voxelMatrix.append(voxelFlat[index:index+self.nEpochs])
            
            voxelsMatrix.append(np.transpose(voxelMatrix))
        
        self.__voxelsMatrix = np.array(voxelsMatrix)
    
    ### calculating the voxel correlation score
    def setVoxelsScore(self):
        if len(self.__voxelsMatrix) > 0:
            
            # find correlation between pairs
            vCorrArray = list()
            for v in range(self.__voxelsMatrix.shape[0]):
                voxel = self.__voxelsMatrix[v]
                corr = 0.0
                for pair in self.__pairs:
                    corr += np.corrcoef(voxel[pair[0]], voxel[pair[1]])[0, 1]
                
                vCorrArray.append(corr/len(self.__pairs))
            
            self.__voxelsScore = np.array(vCorrArray)
            
        else:
            print('Voxel Matrix is not generated')
    
    ### find all possible pairs between epochs
    def __findPossiblePairs(self):
        pairs = []
        for l in itertools.combinations(range(self.nEpochs), 2):
            pairs.append(l)
        self.__pairs = pairs
    
    ### get the voxels score
    def getVoxelsScore(self):
        return self.__voxelsScore
