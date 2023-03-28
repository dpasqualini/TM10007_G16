import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from load_sets import load_D_set
from self_made import return_X_y, removal_zero_var

from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif

D_s = load_D_set()
X_d, y_d = return_X_y(D_s)

# define feature selection
selector = SelectKBest(f_classif, k=1)
selector.fit_transform(X_d, y_d)

scores = -np.log10(selector.pvalues_)
X_indices = np.arange(X_d.shape[-1])

plt.figure(1)
plt.clf()
plt.bar(X_indices, scores, width=0.2)
plt.title("Feature univariate score")
plt.xlabel("Feature number")
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
plt.show()