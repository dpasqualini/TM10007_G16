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




# Fit a simple LDA without feature selection and plot an ROC curve
clf = LDA()
clf.fit(X_tr_norm, y_tr1.to_numpy().reshape(-1,))
y_score = clf.predict_proba(X_tr_norm)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

le = LabelEncoder()
y_tr1_bin = le.fit_transform(y_tr1.to_numpy().reshape(-1,))
fpr,tpr,thesholds = roc_curve(y_tr1_bin, y_score[:, 1])

feature_sizes = [40, 80, 120, 160]

plt.figure(figsize=(8,8))
for size in feature_sizes:
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator=model, n_features_to_select=115)
    rfe.fit(X_tr_norm, y_tr1.to_numpy().reshape(-1,))
    sel_features = X_tr_norm.columns[rfe.support_]
    X_tr_norm_sel= X_tr_norm.loc[:, sel_features]
    # Fit the LDA on selected features
    clf_sel = LDA()
    clf_sel.fit(X_tr_norm_sel, y_tr1.to_numpy().reshape(-1,))
    y_score_sel = clf.predict_proba(X_tr_norm_sel)

    n_original = X_tr_norm.shape[1]
    n_selected = X_tr_norm_sel.shape[1]
    print(f"Selected {n_selected} from {n_original} features.")

    fpr_sel,tpr_sel,thesholds_sel = roc_curve(y_tr1_bin, y_score_sel[:, 1])
    roc_auc_sel = roc_auc_score(y_tr1_bin, y_score_sel[:, 1])

    plt.plot(fpr_sel, tpr_sel, label=f"{size} features (AUC = {roc_auc_sel:.2f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve by Feature Selection")
plt.legend(loc="lower right")
plt.show()


X_tr_norm_sel.boxplot(figsize=(10,5))
plt.title('Normalised data scaled by standard scaling after feature selection by RFE')
plt.show()

# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()