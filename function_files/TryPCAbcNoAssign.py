
from self_made import load_Tr_set, normalize_column, PCA_info_notransform, PCA_plot, PCA_transform, PCA_transform_overview, scree_plot, factor_loadings, classifiers
import pandas as pd
import os
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier # Not regressor as it is for continues variables

### in selfmad

# def PCA_info_notransform(X):
#     # Covariance matrix
#     covar_matrix = PCA(n_components = len(X))
#     covar_matrix.fit(X)
#     eigenvalues = covar_matrix.explained_variance_

#     variance = covar_matrix.explained_variance_ratio_
#     var = np.cumsum(np.round(variance, decimals=3)*100)

#     return variance, var

# def PCA_plot(var):
#     plt.ylabel('% Variance Explained')
#     plt.xlabel('# of Features')
#     plt.title('PCA Variance Explained')
#     plt.ylim(min(var), 100.5)
#     plt.style.context('seaborn-whitegrid')
#     plt.axhline(y=90, color='r', linestyle='--')
#     plt.plot(var)
#     plt.show()
    
#     for j, val in enumerate(var):
#         if val > 90:
#             print("The index of the first number that is bigger than 90% is:", j)
#             n_comp = j+1
#             break
#     return n_comp

# def PCA_transform(X_tr_normalized, n_comp):
#     pca = PCA(n_components=n_comp) # as index starts at 0 the number of components is one higher
#     X_pca = pca.fit_transform(X_tr_normalized)

#     eigenvalues = pca.explained_variance_

#     variance = pca.explained_variance_ratio_
#     var = np.cumsum(np.round(variance, decimals=3)*100)

#     return pca, X_pca, variance, var

# def PCA_transform_overview(variance, var):
#     n_components = len(variance)
#     idx = np.arange(n_components)+1

#     df_explained_variance = pd.DataFrame([variance, var], 
#                                         index=['explained variance', 'cumulative'], 
#                                         columns=idx).T

#     mean_explained_variance = df_explained_variance.iloc[:,0].mean() # calculate mean explained variance

#     # DISPLAY info about PCs
#     print('PCA Overview')
#     print('='*40)
#     print("Total: {} components".format(n_components))
#     print('-'*40)
#     print('Mean explained variance:', round(mean_explained_variance,3))
#     print('-'*40)
#     print(df_explained_variance)
#     print('-'*40)
#     return df_explained_variance, mean_explained_variance, idx

# def scree_plot(df_explained_variance, mean_explained_variance, idx):
#     limit_df = 15 # change this number if you want to have more or less PCA components visualized
#     df_explained_variance_limited = df_explained_variance.iloc[:limit_df,:]

#     #make scree plot
#     fig, ax1 = plt.subplots(figsize=(15,6))

#     ax1.set_title('Explained variance across principal components', fontsize=14)
#     ax1.set_xlabel('Principal component', fontsize=12)
#     ax1.set_ylabel('Explained variance', fontsize=12)

#     ax2 = sns.barplot(x=idx[:limit_df], y='explained variance', data=df_explained_variance_limited, palette='summer')
#     ax2 = ax1.twinx()
#     ax2.grid(False)

#     ax2.set_ylabel('Cumulative', fontsize=14)
#     ax2 = sns.lineplot(x=idx[:limit_df]-1, y='cumulative', data=df_explained_variance_limited, color='#fc8d59')

#     ax1.axhline(mean_explained_variance, ls='--', color='#fc8d59') #plot mean
#     ax1.text(-.8, mean_explained_variance+(mean_explained_variance*.05), "average", color='#fc8d59', fontsize=14) #label y axis

#     max_y1 = max(df_explained_variance_limited.iloc[:,0])
#     max_y2 = max(df_explained_variance_limited.iloc[:,1])
#     ax1.set(ylim=(0, max_y1+max_y1*.1))
#     ax2.set(ylim=(0, max_y2+max_y2*.1))
    
#     return plt.show()

# def factor_loadings(X, pca, df_explained_variance):
#     ############### DEFINE PARAMETERS 
#     selected_features = X.columns
#     top_k = 3
#     #select data based on percentile (top_q) or top-k features
#     top_q = .50
#     top_pc = 3
#     ###############

#     # PCA factor loadings
#     df_c = pd.DataFrame(pca.components_, columns=selected_features).T

#     print("Factor Loadings for the 1. component \n(explains {0:.2f} of the variance)".format(df_explained_variance.iloc[0,0]))
#     print('='*40,'\n')
#     print('Top {} highest'.format(top_k))
#     print('-'*40)
#     print(df_c.iloc[:,0].sort_values(ascending=False)[:top_k], '\n')

#     print('Top {} lowest'.format(top_k))
#     print('-'*40)
#     print(df_c.iloc[:,0].sort_values()[:top_k])

#     # Plot heatmap
#     size_yaxis = round(X[selected_features].shape[1] * 0.5)
#     # Create a mask of values above 0.025
#     mask = df_c > 0.025

#     fig, ax = plt.subplots(figsize=(8,size_yaxis))
#     sns.heatmap(df_c.iloc[:,:top_pc], annot=True, cmap="YlGnBu", ax=ax)
#     return plt.show()

# def classifiers(clsfs, X_pca, y_train):
#     # Loop over each classifier and fit the model, make predictions and compute metrics
#     for clf in clsfs:
#         clf.fit(X_pca, y_train)
#         y_pred = clf.predict(X_pca)
#         accuracy = metrics.accuracy_score(y_train, y_pred)
#         F1 = metrics.f1_score(y_train, y_pred)
#         precision = metrics.precision_score(y_train, y_pred)
#         recall = metrics.recall_score(y_train, y_pred)
#         print(type(clf).__name__)
#         print('Accuracy:', accuracy)
#         print('F1:', F1)
#         print('Precision:', precision)
#         print('Recall:', recall)
#         print()
#     return

# ###

# Load the data
X_train, y_train = load_Tr_set(1)
y_train = y_train['label'].replace({'benign': 0, 'malignant': 1})
y_train = y_train.to_numpy()

# Normalize the data
X_train_n = X_train.apply(normalize_column)
X_train_n = X_train_n.to_numpy()

# To get info about whole dataset in PCA
selected_features = X_train_n
variance1, var1 = PCA_info_notransform(X_train_n, selected_features)
n_comp = PCA_plot(var1)
# To get the new X_pca dataset
pca, X_pca, variance, var = PCA_transform(X_train_n, n_comp)
df_explained_variance, mean_explained_variance, idx = PCA_transform_overview(variance, var)
scree_plot(df_explained_variance, mean_explained_variance, idx) #can be done for every component, now only component 1
factor_loadings(X_train, pca, df_explained_variance) #heatmap should be improved after better features selection 

# Define a list of classifiers
clsfs = [LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), GaussianNB(),
         LogisticRegression(), SGDClassifier(), KNeighborsClassifier(), 
         RandomForestClassifier(), DecisionTreeClassifier(), svm.SVC()]

classifiers(clsfs, X_pca, y_train)






