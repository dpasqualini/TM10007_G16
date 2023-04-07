import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from PCA_tryB import X_train_psig_nn

from self_made import return_X_y, load_D_set, normalize_column

D_s = load_D_set()
X_d, y_d = return_X_y(D_s)

print(X_d)

X_d_normalized = X_d.apply(normalize_column)
print(X_d_normalized)

covar_matrix = PCA(n_components = len(X_d_normalized))
covar_matrix.fit(X_d_normalized)
plt.ylabel('Eigenvalues')
plt.xlabel('# of Features')
plt.title('PCA Eigenvalues')
plt.ylim(0, max(covar_matrix.explained_variance_))
plt.style.context('seaborn-whitegrid')
plt.axhline(y=1, color='r', linestyle='--')
plt.plot(covar_matrix.explained_variance_)
plt.show()


print(covar_matrix.explained_variance_[0:15])
print(f"From index 4 the PCA eigenvalue becomes lower than 1, namely: {covar_matrix.explained_variance_[4]}")

variance = covar_matrix.explained_variance_ratio_
var = np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Variance Explained')
plt.ylim(min(var), 100.5)
plt.style.context('seaborn-whitegrid')
plt.axhline(y=90, color='r', linestyle='--')
plt.plot(var)
plt.show()

print(var[30:33])
print(f"From index 15 the PCA var becomes higher than 90%, namely: {var[31]}")