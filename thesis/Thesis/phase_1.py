import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from random import randint

def train_test_split(split_part):
    og_data = np.load('og_data.npy')
    labels = np.load('labels.npy')
    
    # Label encoder
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    x_train, x_test, y_train, y_test = [], [], [], []

    smin = 0
    smax = 1
    x = og_data
    xmax, xmin = x.max(), x.min()
    norm_data = (x - xmin)*(smax - smin)/(xmax - xmin) + smin

    # Sorting the data
    sorted_index = np.argsort(labels)
    sorted_labels = labels[sorted_index]
    sorted_og_data = og_data[sorted_index]
    sorted_norm_data = norm_data[sorted_index]

    # split the data into train and test
    test_random_indices = []
    for i in range(int(sorted_labels.shape[0]/split_part)):
        index = i*split_part + randint(0, split_part - 1)
        test_random_indices.append(index)

    x_train = np.delete(sorted_norm_data, test_random_indices, 0)
    x_test = sorted_norm_data[test_random_indices]
    y_train = np.delete(sorted_labels, test_random_indices)
    y_test = sorted_labels[test_random_indices]
    
    return x_train, x_test, y_train, y_test

clf = LinearSVC()
x_train, x_test, y_train, y_test = train_test_split(6)

print(x_train.shape)
#clf.fit(x_train, y_train)
