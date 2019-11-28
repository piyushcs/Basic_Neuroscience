from fmriDataset import fmriDataset
from voxelCorr import voxelCorr

dataset = fmriDataset()

X, Y = dataset.getData()

X.shape

train_index = [0, 1, 2, 3, 4]
test_index = [5]

xTrain = X[train_index]

shape = xTrain.shape

xTrain = xTrain.reshape(shape[0]*shape[1], shape[2])

vCorr = voxelCorr(xTrain, 5)

voxelScore = vCorr.getVoxelsScore()
