
# coding: utf-8

# In[1]:


import numpy as np


# In[52]:


from sklearn.svm import LinearSVC
import sklearn
from sklearn.linear_model import LinearRegression
from scipy.io import loadmat
from sklearn import preprocessing
from random import randint


# In[53]:


def train_test_split(split_part):
    og_data = np.load('og_data.npy')
    labels = np.load('labels.npy')
    
    # Label encoder
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    x_train, x_test, y_train, y_test = [], [], [], []

    smin = -1
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
    for i in range(sorted_labels.shape[0]/split_part):
        index = i*split_part + randint(0, split_part - 1)
        test_random_indices.append(index)

    x_train = np.delete(sorted_norm_data, test_random_indices, 0)
    x_test = sorted_norm_data[test_random_indices]
    y_train = np.delete(sorted_labels, test_random_indices)
    y_test = sorted_labels[test_random_indices]
    
    return x_train, x_test, y_train, y_test


# In[54]:


def get_verbs():
    layer_2 = np.load('verb_vectors.npy')
    labels = np.load('verb_labels.npy')
    
    x_train, x_test, y_train, y_test = train_test_split(6)
    
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


# In[129]:


x_train, x_test, y_train, y_test, layer_2_train, layer_2_test = get_verbs()


# In[130]:


print x_train.shape, y_train.shape, layer_2_train.shape, layer_2_test.shape


# In[ ]:


regr = []

for i in range(25):
    print(i)
    regr.append(LinearRegression())
    regr[i].fit(x_train, layer_2_train[:, i])

print regr


# In[100]:


output = list()

for i in range(len(regr)):
    output.append(regr[i].predict(x_test))

output = np.transpose(output)


# In[128]:


clf = LinearSVC()
layer_2 = np.load('verb_vectors.npy')
labels = np.load('verb_labels.npy')
le = preprocessing.LabelEncoder()
le.fit(labels)
labels = list(le.transform(labels))

clf.fit(layer_2, labels)

decision_lbl = clf.decision_function(output)[0]

for i in [1, 3, 5, 10]:
    predicted_correct = 0
    for k in range(y_test.shape[0]):
        o = output[k]
        o = o.reshape(1, -1)
        decision_lbl = clf.decision_function(o)[0]
        top_class = i
        top_n = np.argsort(decision_lbl)[::-1][:top_class]
        if y_test[k] in top_n:
            predicted_correct += 1.0/60
    print predicted_correct


# In[107]:


predicted_correct = 0

predicted_labels = clf.predict(output)
for i in range(y_test.shape[0]):
    if predicted_labels[i] == y_test[i]:
        predicted_correct += 1


# In[103]:


predicted_correct

