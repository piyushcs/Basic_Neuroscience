#### Abstract

Predicting noun labels using fMRI images is a difficult task, due to high dimension of fMRI images and it is also costly to obtain large datasets to overcome this curse of dimensionality. In the dataset of a single participant, to predict 60 different classes, only 360 images are available, 6 images per class. Using liblinear SVM on original dataset 21.11 of labels can be predicted correctly but with high variance.

A significant advantage for verbs compared to nouns in picture-naming tests has been observed previously, so rather than directly predicting noun, a noun can be represented as a verb vector and using Multi-output regression on fMRI images verb feature vector can be obtained. This approach doesn’t improve the result and gives similar results as the baseline results. Using Google news word vector also doesn’t improve the outcomes. Creating custom properties for labels, for example, animate, natural to divide the original label set into different groups is also implemented using building a binary classifier per property. Using this approach, labels can be divided into groups, but we cannot classify all the noun labels.

Using standard feature selection methods, for example, PCA doesn’t improve the results. Using Autoencoder with Logistic regression on a 3D representation of fMRI images improve the Top1 accuracy, but variance remains very high. This gave us motivation that careful feature selection can be helpful for our task. Voxel pairwise correlation-based feature selection for the same noun across multiple epochs can be used in feature selection. Choosing the number of voxels remains a problem. To avoid brute force a region-based approach is used, occipital lobe, visual processing center can be used to select the number of features.

For example, Voxel-400 (choosing top 400 highly correlated voxels), all voxels end up in occipital lobe in the 3D representation of fMRI image. Top1, Top3, Top5 and Top10 accuracy improved significantly using liblinear SVM. 3D-CNN and Logistic Regression also show improved results using these features. Variance also decrease when choosing region based approach. When choosing a large number of voxels, variance increases, and accuracy drops, because voxels end up outside occipital lobe.

#### Results

| Model used for training  | Classification Accuracy Mean | Variance |
| ------------------------ | ---------------------------: | -------: |
| Liblinear SVM            | 21.11                        | 4.0      |
| Logistic regression      | 17.22                        | 5.5      |
| CMU Verb vectors         | 13.05                        | 3.0      |
| Google word vectors      | 26.95                        | 4.2      |
| Autoencoder              | 20.28                        | 3.0      |
| Voxel-800                | 51.94                        | 6.2      |
| Voxel-1000               | 51.38                        | 6.0      |
| CNN + voxel              | 35.72                        | 5.6      |
| wordvec + voxel          | 31.39                        | 3.9      |
