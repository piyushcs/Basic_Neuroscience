from dataset import fmriDataset, dataSplit
from preprocessing import voxelCorr, verbVectors
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

import argparse
import os
import sys

import warnings
warnings.filterwarnings("ignore")

arg_parser = argparse.ArgumentParser(description='specify model')

arg_parser.add_argument('--model',
                       action='store',
                       type=str,
                       help='specify model for example verb, voxel, baseline, gvmodel etc.',
                       required=True)

arg_parser.add_argument('--crossValidation',
                       action='store',
                       type=bool,
                       help='cross validate or not',
                       default=False)

args = arg_parser.parse_args()

if args.crossValidation:
    print('not implemented yet')
    sys.exit(1)

# Loading the dataset
dataset = fmriDataset.fmriDataset()

print("Loading dataset")
X, Y = dataset.getData()
print(X.shape)

print()

print("Test Train data split")
dataSplit = dataSplit.dataSplit(X, Y)
xTrain, yTrain, xTest, yTest = dataSplit.trainTestSplit(itr=4)
print(xTrain.shape)
print(xTest.shape)
print()

if args.model == 'verb':
    print("Semantic verb vectors model")
    sv = verbVectors.semanticVectors()
    
    nLabels = sv.nounLabels
    verbVectors = sv.verbVectors
    
    verbVectorsTrain = []
    for yl in yTrain:
        i = nLabels.index(yl)
        verbVectorsTrain.append(verbVectors[i])
    
    verbVectorsTest = []
    for yl in yTest:
        i = nLabels.index(yl)
        verbVectorsTest.append(verbVectors[i])
    
    print()
    print("Linear Regression model")
    print("Training layer 1")
    regr = LinearRegression()
    regr.fit(xTrain, verbVectorsTrain)
    
    output = regr.predict(xTest)
    
    print()
    print("Linear SVC")
    print("Training layer 2")
    clf = LinearSVC()
    clf.fit(verbVectors, nLabels)
    print()
    
    print("Test data Score")
    print(clf.score(output, yTest))
    print()

elif args.model == 'voxel':
    
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

elif args.model == 'baseline':
    
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

elif args.model == 'gvmodel':
    
    nLabels = np.load('preprocessing/google_lbls.npy')
    gverbVectors = np.load('preprocessing/google_word_vec.npy')
    
    nLabels = nLabels.tolist()
    
    gverbVectorsTrain = []
    for yl in yTrain:
        i = nLabels.index(yl)
        gverbVectorsTrain.append(gverbVectors[i])
    
    gverbVectorsTest = []
    for yl in yTest:
        i = nLabels.index(yl)
        gverbVectorsTest.append(gverbVectors[i])
    
    print()
    print("Linear Regression model")
    print("Training layer 1")
    regr = LinearRegression()
    regr.fit(xTrain, gverbVectorsTrain)
    
    output = regr.predict(xTest)
    
    print()
    print("Linear SVC")
    print("Training layer 2")
    clf = LinearSVC()
    clf.fit(gverbVectors, nLabels)
    print()
    
    print("Test data Score")
    print(clf.score(output, yTest))
    print()
    
    
else:
    print('Check models available')