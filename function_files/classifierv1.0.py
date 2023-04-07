import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier,  RandomForestRegressor  # For classification and regression tasks
from sklearn.metrics import f1_score

from self_made import load_Tr_set, normalize_column

# General packages
from sklearn import datasets as ds
# Metrics

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn import metrics


# Load the data
X_train, y_train = load_Tr_set(1)
y_train = y_train['label'].replace({'benign': 0, 'malignant': 1})
y_train = y_train.to_numpy()

# Normalize the data
X_train_n = X_train.apply(normalize_column)
X_train_n = X_train_n.to_numpy()

# Perform PCA
pca = PCA(n_components=33)
X_pca = pca.fit_transform(X_train_n)

# Visualize the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
#plt.show()

X_ben = X_pca[y_train == 0]
X_mal = X_pca[y_train == 1]

# Calculate the covariance
covariance = np.cov(X_ben, X_mal)[0][1]

# Print the covariance
print("Covariance:", covariance)

# Calculate the covariance matrix
cov_matrix_ben= np.cov(X_ben, rowvar=True)
cov_matrix_mal = np.cov(X_mal, rowvar=True)

# Print the covariance matrix
print("Cov_matrix ben", cov_matrix_ben)
print("Cov_matrix mal", cov_matrix_mal)

# Define a list of classifiers
clsfs = [LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), GaussianNB(),
         LogisticRegression(), SGDClassifier(), KNeighborsClassifier()]

# Loop over each classifier and fit the model, make predictions and compute metrics
for clf in clsfs:
    clf.fit(X_pca, y_train)
    y_pred = clf.predict(X_pca)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    F1 = metrics.f1_score(y_train, y_pred)
    precision = metrics.precision_score(y_train, y_pred)
    recall = metrics.recall_score(y_train, y_pred)
    print(type(clf).__name__)
    print('Accuracy:', accuracy)
    print('F1:', F1)
    print('Precision:', precision)
    print('Recall:', recall)
    print()