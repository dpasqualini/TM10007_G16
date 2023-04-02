import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from self_made import return_X_y, removal_zero_var, load_Tr_set, normalize_column, load_Va_set

# X_train, y_train = load_Tr_set(1)
# y_train = y_train.to_numpy()
# X_val, y_val = load_Va_set(1)
# y_val = y_val.to_numpy()

# X_train_n = X_train.apply(normalize_column)
# X_val_n = X_val.apply(normalize_column)

# X_train_n = X_train_n.to_numpy()
# X_val_n = X_val_n.to_numpy()


# cov_mat = np.cov(X_train_n.T)
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# tot = sum(eigen_vals)
# var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)

from sklearn.decomposition import PCA

# Load the data
X_train, y_train = load_Tr_set(1)
y_train = y_train['label'].replace({'benign': 0, 'malignant': 1})
y_train = y_train.to_numpy()

# Normalize the data
X_train_n = X_train.apply(normalize_column)
X_train_n = X_train_n.to_numpy()

# Perform PCA
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X_train_n)

# Visualize the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# Plot the different features for malignant and benign tumors
fig, axes = plt.subplots(5, 2, figsize=(10, 20))
for i, ax in enumerate(axes.ravel()):
    _, bins = np.histogram(X_train.iloc[:, i], bins=50)
    ax.hist(X_train.iloc[y_train == 0, i], bins=bins, color='b', alpha=0.5, density=True)
    ax.hist(X_train.iloc[y_train == 1, i], bins=bins, color='r', alpha=0.5, density=True)
    ax.set_title(X_train.columns[i])
    ax.set_yticks([])
    ax.legend(['Benign', 'Malignant'])

plt.tight_layout()
plt.show()
