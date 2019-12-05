from dataset import fmriDataset, dataSplit
from preprocessing import voxelCorr
import numpy as np
from sklearn.svm import LinearSVC

dataset = fmriDataset.fmriDataset()

print("Loading dataset")
X, Y = dataset.getData()
print(X.shape)

print()

print("Test Train data split")
dataSplit = dataSplit.dataSplit(X, Y)
xTrain, yTrain, xTest, yTest = dataSplit.trainTestSplit(1)
print(xTrain.shape)
print(xTest.shape)

print()
print("Baseline:")
print("================")
print("Training model")
clf = LinearSVC()
clf.fit(xTrain, yTrain)

print('.')
print("Model Training finished")

print()

print("Test data Score")
print(clf.score(xTest, yTest))
print()

print()

print("Voxel 800 model:")
print("================")
print("Calculating voxel correlation scores using training dataset")
vCorr = voxelCorr.voxelCorr(xTrain, 5)
voxelScore = vCorr.getVoxelsScore()
print("Max voxel correlation score", max(voxelScore))

print()

print("Selecting voxels with high voxel correlation scores")
sortedIndex = np.argsort(voxelScore)[::-1]
xTrain = xTrain[:, sortedIndex[:800]]
xTest = xTest[:, sortedIndex[:800]]

print()

print("Training model")
clf = LinearSVC()
clf.fit(xTrain, yTrain)

print('.')
print("Model Training finished")

print()

print("Test data Score")
print(clf.score(xTest, yTest))
print()
