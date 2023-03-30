import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from load_sets import load_D_set
from self_made import return_X_y, removal_zero_var

from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import kendalltau
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression

D_s = load_D_set()
X_d, y_d = return_X_y(D_s)
df_filter = removal_zero_var(X_d)


# define feature selection by univariate with f_classiff (lineair data with numerical input and categorial output)
selector = SelectKBest(f_classif, k=2)
selector.fit_transform(df_filter, y_d)

mask = selector.get_support()
selected_features = np.array(df_filter.columns)[mask]
print(selected_features)

# wrapper function for kendalltau that returns p-values

# def kendalltau_pvalues(df_filter, y):
#     pvals = []
#     for feature in df_filter.T:
#         tau, pval = kendalltau(feature, y)
#         pvals.append(pval)
#     return pvals

# y = y_d.values.ravel()
# # using SelectKBest with kendalltau wrapper
# selector = SelectKBest(score_func=kendalltau_pvalues, k=3)
# X_new = selector.fit_transform(df_filter, y)
# print(X_new)

# printing the individual correlation of the features with the outcome y_d 
scores = -np.log10(selector.pvalues_)
X_indices = np.arange(df_filter.shape[-1])


plt.figure(1)
plt.clf()
plt.bar(X_indices, scores, width=0.2)
plt.title("Feature univariate score")
plt.xlabel("Feature number")
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
plt.show()

