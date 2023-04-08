import pandas as pd
import os
import numpy as np
import seaborn as sns
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
pca = PCA(n_components=31) # 33 is choosen as higher than 90 percent variance suggests taking n_components of 33
X_pca = pca.fit_transform(X_train_n)

# Visualize the results
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# #plt.show()

# get basic info
n_components = len(pca.explained_variance_ratio_)
explained_variance = pca.explained_variance_ratio_
cum_explained_variance = np.cumsum(explained_variance)
idx = np.arange(n_components)+1

df_explained_variance = pd.DataFrame([explained_variance, cum_explained_variance], 
                                     index=['explained variance', 'cumulative'], 
                                     columns=idx).T

mean_explained_variance = df_explained_variance.iloc[:,0].mean() # calculate mean explained variance

# DISPLAY info about PCs
print('PCA Overview')
print('='*40)
print("Total: {} components".format(n_components))
print('-'*40)
print('Mean explained variance:', round(mean_explained_variance,3))
print('-'*40)
print(df_explained_variance)
print('-'*40)

#limit plot to x PC
limit = int(input("Limit scree plot to nth component (0 for all) > "))
if limit > 0:
    limit_df = limit
else:
    limit_df = n_components

df_explained_variance_limited = df_explained_variance.iloc[:limit_df,:]

#make scree plot
fig, ax1 = plt.subplots(figsize=(15,6))

ax1.set_title('Explained variance across principal components', fontsize=14)
ax1.set_xlabel('Principal component', fontsize=12)
ax1.set_ylabel('Explained variance', fontsize=12)

ax2 = sns.barplot(x=idx[:limit_df], y='explained variance', data=df_explained_variance_limited, palette='summer')
ax2 = ax1.twinx()
ax2.grid(False)

ax2.set_ylabel('Cumulative', fontsize=14)
ax2 = sns.lineplot(x=idx[:limit_df]-1, y='cumulative', data=df_explained_variance_limited, color='#fc8d59')

ax1.axhline(mean_explained_variance, ls='--', color='#fc8d59') #plot mean
ax1.text(-.8, mean_explained_variance+(mean_explained_variance*.05), "average", color='#fc8d59', fontsize=14) #label y axis

max_y1 = max(df_explained_variance_limited.iloc[:,0])
max_y2 = max(df_explained_variance_limited.iloc[:,1])
ax1.set(ylim=(0, max_y1+max_y1*.1))
ax2.set(ylim=(0, max_y2+max_y2*.1))

plt.show()

# ### Factor loadings!! ###
selected_features = X_train.columns
print(selected_features)

############### DEFINE PARAMS
top_k = 3
#select data based on percentile (top_q) or top-k features
top_q = .50
top_pc = 3
###############

# PCA factor loadings
df_c = pd.DataFrame(pca.components_, columns=selected_features).T

print("Factor Loadings for the 1. component \n(explains {0:.2f} of the variance)".format(df_explained_variance.iloc[0,0]))
print('='*40,'\n')
print('Top {} highest'.format(top_k))
print('-'*40)
print(df_c.iloc[:,0].sort_values(ascending=False)[:top_k], '\n')

print('Top {} lowest'.format(top_k))
print('-'*40)
print(df_c.iloc[:,0].sort_values()[:top_k])

# Plot heatmap
size_yaxis = round(X_train[selected_features].shape[1] * 0.5)
# Create a mask of values above 0.025
mask = df_c > 0.025

fig, ax = plt.subplots(figsize=(8,size_yaxis))
sns.heatmap(df_c.iloc[:,:top_pc], annot=True, cmap="YlGnBu", ax=ax)
plt.show()

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