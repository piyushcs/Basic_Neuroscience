from dataset import fmriDataset, dataSplit
from preprocessing import voxelCorr, verbVectors
from models import verbModels, voxelModels
import numpy as np

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
                       default="voxel")
                       # required=True)

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
dataset = fmriDataset.fmriDataset(multiDimention=True)

print("Loading dataset")
X, Y = dataset.getData()
print(X.shape)

print()

print("Test Train data split")
dataSplit = dataSplit.dataSplit(X, Y)
xTrain, yTrain, xTest, yTest = dataSplit.trainTestSplit(itr=3)
print(xTrain.shape)
print(xTest.shape)
print()

if args.model == 'verb':
    print("Semantic verb vectors model")
    
    try:
        nLabels = np.load('preprocessing/cmu_nLabels.npy')
        verbVectors = np.load('preprocessing/cmu_verbVectors.npy')
        nLabels = nLabels.tolist()

    except:
        sv = verbVectors.semanticVectors()    
        nLabels = sv.nounLabels
        verbVectors = sv.verbVectors
        np.save("preprocessing/cmu_nLabels.npy", nLabels)
        np.save("preprocessing/cmu_verbVectors.npy", verbVectors)
    
    vm = verbModels.verbModel(xTrain, yTrain, xTest, yTest)
    vm.parseVerbVectors(nLabels, verbVectors)
    vm.layer1()
    vm.layer2(nLabels, verbVectors)
    print()

    print(vm.score)

elif args.model == 'voxel':
    
    print("Voxel 1000 model:")
    print("================")
    print("Calculating voxel correlation scores using training dataset")
    vCorr = voxelCorr.voxelCorr(xTrain, 5)
    voxelScore = vCorr.getVoxelsScore()
    print("Max voxel correlation score", max(voxelScore))

    print()

    print("Training model")
    vm = voxelModels.voxel(xTrain, yTrain, xTest, yTest)
    vm.transform(voxelScore)
    vm.train()

    print('.')
    print("Model Training finished")

    print()

    print("Test data Score")
    print(vm.score)
    print()

elif args.model == 'baseline':
    
    print("Baseline:")
    print("================")
    print("Training model")
    baseline = voxelModels.baseline(xTrain, yTrain, xTest, yTest)
    baseline.train()
    
    print('.')
    print("Model Training finished")

    print()

    print("Test data Score")
    print(baseline.score)
    print()

elif args.model == 'gvmodel':
    
    nLabels = np.load('preprocessing/google_lbls.npy')
    gverbVectors = np.load('preprocessing/google_word_vec.npy')
    nLabels = nLabels.tolist()

    gm = verbModels.verbModel(xTrain, yTrain, xTest, yTest)
    gm.parseVerbVectors(nLabels, gverbVectors)
    gm.layer1()
    gm.layer2(nLabels, gverbVectors)
    
    print(gm.score)
    
    
else:
    print('Check models available')
