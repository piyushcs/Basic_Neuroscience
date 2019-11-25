
# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(28)
import itertools
import random
random.seed(28)
pairs = []
for l in itertools.combinations(range(5), 2):
    pairs.append(l)


# In[2]:


from sklearn.svm import LinearSVC
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from scipy.io import loadmat
from sklearn import preprocessing
from random import randint
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA


# In[3]:


def train_test_split(indices):
    og_data = np.load('og_data1.npy')
    labels = np.load('labels1.npy')
    
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

    # print test_random_indices
    x_train = np.delete(sorted_norm_data, indices, 0)
    x_test = sorted_norm_data[indices]
    y_train = np.delete(sorted_labels, indices)
    y_test = sorted_labels[indices]
    
    return x_train, x_test, y_train, y_test


# In[4]:


def plot(img):
#     x_train, x_test, y_train, y_test = train_test_split(6)
    fig = plt.figure(figsize=(14, 12))
    columns = 5
    rows = 5
    for i in range(columns*rows):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img[:,:, i])
        plt.axis('off')
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig("/Users/piyush.kumar/Desktop/demo_data/airplane_voxe400.png", bbox_inches='tight', frameon=False)
# plot(og)


# In[5]:


def get_verbs(indices):
    layer_2 = np.load('google_word_vec.npy')
    labels = np.load('google_lbls.npy')
    
    x_train, x_test, y_train, y_test = train_test_split(indices)
    
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = list(le.transform(labels))
    
    layer_2_train = []
    for yl in y_train:
        i = labels.index(yl)
        layer_2_train.append(layer_2[i])

    layer_2_test = []
    for yt in y_test:
        i = labels.index(yt)
        layer_2_test.append(layer_2[i])

    return x_train, x_test, y_train, y_test, np.array(layer_2_train), np.array(layer_2_test)

def get_mat(x_train):
    voxel_corr_matrix = []
    for i in range(x_train.shape[1]):
        voxel = x_train[:, i]
        pre_corr_matrix = []
        for j in range(60):
            k = j*5
            pre_corr_matrix.append(voxel[k:k+5])

        voxel_corr_matrix.append(np.transpose(pre_corr_matrix))

    voxel_corr_matrix = np.asarray(voxel_corr_matrix)
    
    return voxel_corr_matrix

# seq_indices = []
# for i in range(60):
#     index = np.array([0, 1, 2, 3, 4, 5])
#     seq_indices.append(index + 6*i)

# np.save('seq', seq_indices)

# print seq_indices

# shuffle_indices = []
# for i in range(60):
#     index = np.array([0, 1, 2, 3, 4, 5])
#     random.shuffle(index)
#     shuffle_indices.append(index + 6*i)

# shuffle_indices = np.asarray(shuffle_indices)
# np.save('shuffle', shuffle_indices)

# print shuffle_indices


# for i in range(1):
#     x_train, x_test, y_train, y_test = train_test_split(seq_indices[:, i])
#     print x_train.shape, x_test.shape


# LogisticRegressionr = []
# for i in range(6):
#     x_train, x_test, y_train, y_test, layer_2_train, layer_2_test = get_verbs(seq_indices[:, i])

#     voxel_corr_matrix = get_mat(x_train)

#     corr_arr = []
#     for i in range(voxel_corr_matrix.shape[0]):
#         voxel = voxel_corr_matrix[i]
#         corr = 0.0
#         for pair in pairs:
#             corr += np.corrcoef(voxel[pair[0]], voxel[pair[1]])[0, 1]
#         corr_arr.append(corr/len(pairs))

#     c_a = np.asarray(corr_arr)
#     sorted_index = np.argsort(c_a)[::-1]

#     x_train = x_train[:, sorted_index[:10000]]
#     x_test = x_test[:, sorted_index[:10000]]

#     regr = LinearRegression()
#     regr.fit(x_train, layer_2_train)

#     output = regr.predict(x_test)

#     clf = LinearSVC()
#     layer_2 = np.load('google_word_vec.npy')
#     labels = np.load('google_lbls.npy')
#     le = preprocessing.LabelEncoder()
#     le.fit(labels)
#     labels = list(le.transform(labels))

#     clf.fit(layer_2, labels)

#     results_top = []
#     for i in [1, 3, 5, 10]:
#         predicted_correct = 0
#         for k in range(y_test.shape[0]):
#             o = output[k]
#             o = o.reshape(1, -1)
#             decision_lbl = clf.decision_function(o)[0]
#             top_class = i
#             top_n = np.argsort(decision_lbl)[::-1][:top_class]
#             if y_test[k] in top_n:
#                 predicted_correct += 1.0/60
#         results_top.append(predicted_correct)
#     LogisticRegressionr.append(results_top)

def result(nof):

    seq_indices = np.load('seq.npy')
    shuffle_indices = np.load('shuffle.npy')

    LogisticRegressionr = []
    for i in range(6):
        x_train, x_test, y_train, y_test = train_test_split(seq_indices[:, i])

        voxel_corr_matrix = get_mat(x_train)

        corr_arr = []
        for i in range(voxel_corr_matrix.shape[0]):
            voxel = voxel_corr_matrix[i]
            corr = 0.0
            for pair in pairs:
                corr += np.corrcoef(voxel[pair[0]], voxel[pair[1]])[0, 1]
            corr_arr.append(corr/len(pairs))

        c_a = np.asarray(corr_arr)
        sorted_index = np.argsort(c_a)[::-1]

        x_t = x_train[:, sorted_index[:nof]]
        x_ts = x_test[:, sorted_index[:nof]]

        clf = LinearSVC()
        clf.fit(x_t, y_train)

        results_top = []
        for i in [1, 3, 5, 10]:
            predicted_correct = 0
            for k in range(y_test.shape[0]):
                o = x_ts[k]
                o = o.reshape(1, -1)
                decision_lbl = clf.decision_function(o)[0]
                top_class = i
                top_n = np.argsort(decision_lbl)[::-1][:top_class]
                if y_test[k] in top_n:
                    predicted_correct += 1.0/60
            results_top.append(predicted_correct)
        LogisticRegressionr.append(results_top)


    print np.average(LogisticRegressionr, axis=0)
    print np.std(LogisticRegressionr, axis=0)

    LogisticRegressionr = []
    for i in range(6):
        x_train, x_test, y_train, y_test = train_test_split(shuffle_indices[:, i])

        voxel_corr_matrix = get_mat(x_train)

        corr_arr = []
        for i in range(voxel_corr_matrix.shape[0]):
            voxel = voxel_corr_matrix[i]
            corr = 0.0
            for pair in pairs:
                corr += np.corrcoef(voxel[pair[0]], voxel[pair[1]])[0, 1]
            corr_arr.append(corr/len(pairs))

        c_a = np.asarray(corr_arr)
        sorted_index = np.argsort(c_a)[::-1]

        x_t = x_train[:, sorted_index[:nof]]
        x_ts = x_test[:, sorted_index[:nof]]

        clf = LinearSVC()
        clf.fit(x_t, y_train)

        results_top = []
        for i in [1, 3, 5, 10]:
            predicted_correct = 0
            for k in range(y_test.shape[0]):
                o = x_ts[k]
                o = o.reshape(1, -1)
                decision_lbl = clf.decision_function(o)[0]
                top_class = i
                top_n = np.argsort(decision_lbl)[::-1][:top_class]
                if y_test[k] in top_n:
                    predicted_correct += 1.0/60
            results_top.append(predicted_correct)
        LogisticRegressionr.append(results_top)


    print np.average(LogisticRegressionr, axis=0)
    print np.std(LogisticRegressionr, axis=0)

test = [800]

for i in test:
    print 'results for', i
    result(i)
    print ''

# LogisticRegressionr = []
# for i in range(6):
#     x_train, x_test, y_train, y_test = train_test_split(shuffle_indices[:, i])

#     voxel_corr_matrix = get_mat(x_train)

#     corr_arr = []
#     for i in range(voxel_corr_matrix.shape[0]):
#         voxel = voxel_corr_matrix[i]
#         corr = 0.0
#         for pair in pairs:
#             corr += np.corrcoef(voxel[pair[0]], voxel[pair[1]])[0, 1]
#         corr_arr.append(corr/len(pairs))

#     c_a = np.asarray(corr_arr)
#     sorted_index = np.argsort(c_a)[::-1]

#     x_t = x_train[:, sorted_index[:2000]]
#     x_ts = x_test[:, sorted_index[:2000]]

#     clf = LinearSVC()
#     clf.fit(x_t, y_train)

#     results_top = []
#     for i in [1, 3, 5, 10]:
#         predicted_correct = 0
#         for k in range(y_test.shape[0]):
#             o = x_ts[k]
#             o = o.reshape(1, -1)
#             decision_lbl = clf.decision_function(o)[0]
#             top_class = i
#             top_n = np.argsort(decision_lbl)[::-1][:top_class]
#             if y_test[k] in top_n:
#                 predicted_correct += 1.0/60
#         results_top.append(predicted_correct)
#     LogisticRegressionr.append(results_top)


# print np.average(LogisticRegressionr, axis=0)
# print np.std(LogisticRegressionr, axis=0)

# LogisticRegressionr = []
# for i in range(6):
#     x_train, x_test, y_train, y_test = train_test_split(shuffle_indices[:, i])

#     voxel_corr_matrix = get_mat(x_train)

#     corr_arr = []
#     for i in range(voxel_corr_matrix.shape[0]):
#         voxel = voxel_corr_matrix[i]
#         corr = 0.0
#         for pair in pairs:
#             corr += np.corrcoef(voxel[pair[0]], voxel[pair[1]])[0, 1]
#         corr_arr.append(corr/len(pairs))

#     c_a = np.asarray(corr_arr)
#     sorted_index = np.argsort(c_a)[::-1]

#     x_t = x_train[:, sorted_index[:3000]]
#     x_ts = x_test[:, sorted_index[:3000]]

#     clf = LinearSVC()
#     clf.fit(x_t, y_train)

#     results_top = []
#     for i in [1, 3, 5, 10]:
#         predicted_correct = 0
#         for k in range(y_test.shape[0]):
#             o = x_ts[k]
#             o = o.reshape(1, -1)
#             decision_lbl = clf.decision_function(o)[0]
#             top_class = i
#             top_n = np.argsort(decision_lbl)[::-1][:top_class]
#             if y_test[k] in top_n:
#                 predicted_correct += 1.0/60
#         results_top.append(predicted_correct)
#     LogisticRegressionr.append(results_top)


# print np.average(LogisticRegressionr, axis=0)
# print np.std(LogisticRegressionr, axis=0)


# LogisticRegressionr = []
# for i in range(6):
#     x_train, x_test, y_train, y_test = train_test_split(seq_indices[:, i])

#     voxel_corr_matrix = get_mat(x_train)

#     corr_arr = []
#     for i in range(voxel_corr_matrix.shape[0]):
#         voxel = voxel_corr_matrix[i]
#         corr = 0.0
#         for pair in pairs:
#             corr += np.corrcoef(voxel[pair[0]], voxel[pair[1]])[0, 1]
#         corr_arr.append(corr/len(pairs))

#     c_a = np.asarray(corr_arr)
#     sorted_index = np.argsort(c_a)[::-1]

#     x_t = x_train[:, sorted_index[:2000]]
#     x_ts = x_test[:, sorted_index[:2000]]

#     clf = LinearSVC()
#     clf.fit(x_t, y_train)

#     results_top = []
#     for i in [1, 3, 5, 10]:
#         predicted_correct = 0
#         for k in range(y_test.shape[0]):
#             o = x_ts[k]
#             o = o.reshape(1, -1)
#             decision_lbl = clf.decision_function(o)[0]
#             top_class = i
#             top_n = np.argsort(decision_lbl)[::-1][:top_class]
#             if y_test[k] in top_n:
#                 predicted_correct += 1.0/60
#         results_top.append(predicted_correct)
#     LogisticRegressionr.append(results_top)


# print np.average(LogisticRegressionr, axis=0)
# print np.std(LogisticRegressionr, axis=0)

# LogisticRegressionr = []
# for i in range(6):
#     x_train, x_test, y_train, y_test = train_test_split(seq_indices[:, i])

#     voxel_corr_matrix = get_mat(x_train)

#     corr_arr = []
#     for i in range(voxel_corr_matrix.shape[0]):
#         voxel = voxel_corr_matrix[i]
#         corr = 0.0
#         for pair in pairs:
#             corr += np.corrcoef(voxel[pair[0]], voxel[pair[1]])[0, 1]
#         corr_arr.append(corr/len(pairs))

#     c_a = np.asarray(corr_arr)
#     sorted_index = np.argsort(c_a)[::-1]

#     x_t = x_train[:, sorted_index[:3000]]
#     x_ts = x_test[:, sorted_index[:3000]]

#     clf = LinearSVC()
#     clf.fit(x_t, y_train)

#     results_top = []
#     for i in [1, 3, 5, 10]:
#         predicted_correct = 0
#         for k in range(y_test.shape[0]):
#             o = x_ts[k]
#             o = o.reshape(1, -1)
#             decision_lbl = clf.decision_function(o)[0]
#             top_class = i
#             top_n = np.argsort(decision_lbl)[::-1][:top_class]
#             if y_test[k] in top_n:
#                 predicted_correct += 1.0/60
#         results_top.append(predicted_correct)
#     LogisticRegressionr.append(results_top)


# print np.average(LogisticRegressionr, axis=0)
# print np.std(LogisticRegressionr, axis=0)


