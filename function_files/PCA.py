import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from self_made import return_X_y, removal_zero_var, load_Tr_set, normalize_column, load_Va_set

X_train, y_train = load_Tr_set(1)
y_train = y_train.to_numpy()
X_val, y_val = load_Va_set(1)
y_val = y_val.to_numpy()

X_train_n = X_train.apply(normalize_column)
X_val_n = X_val.apply(normalize_column)

X_train_n = X_train_n.to_numpy()
X_val_n = X_val_n.to_numpy()


cov_mat = np.cov(X_train_n.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# # plot explained variances
# print('hello')
# plt.bar(range(len(var_exp)), var_exp, alpha=0.5,
#         align='center', label='individual explained variance')
# print('hello')
# plt.step(range(len(var_exp)), cum_var_exp, where='mid',
#          label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal component index')
# plt.legend(loc='best')
# plt.show()

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))

X_train_n[0].dot(w)

X_train_pca = X_train_n.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for v, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == v][:, 0]), plt.scatter(X_train_pca[y_train == v][:, 1], c=c, label=v, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
