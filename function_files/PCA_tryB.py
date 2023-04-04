import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier,  RandomForestRegressor  # For classification and regression tasks
from sklearn.metrics import f1_score


from self_made import load_Tr_set, normalize_column

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

# Load the data
X_train, y_train = load_Tr_set(1)
y_train = y_train['label'].replace({'benign': 0, 'malignant': 1})
y_train = y_train.to_numpy()

# Normalize the data
X_train_n = X_train.apply(normalize_column)
X_train_n = X_train_n.to_numpy()

# Perform PCA
# pca = PCA(n_components=4)
# X_pca = pca.fit_transform(X_train_n)

# # Visualize the results
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()


# # Plot the different features for malignant and benign tumors
# fig, axes = plt.subplots(5, 2, figsize=(10, 20))
# for i, ax in enumerate(axes.ravel()):
#     _, bins = np.histogram(X_train.iloc[:, i], bins=50)
#     ax.hist(X_train.iloc[y_train == 0, i], bins=bins, color='b', alpha=0.5, density=True)
#     ax.hist(X_train.iloc[y_train == 1, i], bins=bins, color='r', alpha=0.5, density=True)
#     ax.set_title(X_train.columns[i])
#     ax.set_yticks([])
#     ax.legend(['Benign', 'Malignant'])

# plt.tight_layout()
# plt.show()

# Split data into two groups based on target variable
group1 = X_train_n[y_train == 0]
group2 = X_train_n[y_train == 1]

# Perform t-test for each feature
t_scores = []
p_values = []
for i in range(X_train_n.shape[1]):
    t, p = ttest_ind(group1[:, i], group2[:, i])
    t_scores.append(t)
    p_values.append(p)

# Create a DataFrame with t-scores and p-values
results = pd.DataFrame({'feature': X_train.columns, 't_score': t_scores, 'p_value': p_values})

# Filter DataFrame to show only significant features
significant_results = results[results['p_value'] < 0.05]

# Print significant results
print(significant_results)

print(significant_results.iloc[:,0])
print(f"X_train is {X_train}")
X_train_psig = X_train.loc[:, significant_results['feature']]
print(X_train_psig)

X_train_psig_n = X_train_psig.apply(normalize_column)
X_train_psig_nn = X_train_psig_n.to_numpy()

# Perform PCA
pca = PCA(n_components=9)
X_pca = pca.fit_transform(X_train_psig_nn)

# Visualize the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# Plot the different features for malignant and benign tumors
fig, axes = plt.subplots(5, 2, figsize=(10, 20))
for i, ax in enumerate(axes.ravel()):
    _, bins = np.histogram(X_train_psig_n.iloc[:, i], bins=50)
    ax.hist(X_train_psig_n.iloc[y_train == 0, i], bins=bins, color='b', alpha=0.5, density=True)
    ax.hist(X_train_psig_n.iloc[y_train == 1, i], bins=bins, color='r', alpha=0.5, density=True)
    ax.set_title(X_train_psig_n.columns[i])
    ax.set_yticks([])
    ax.legend(['Benign', 'Malignant'])

plt.tight_layout()
plt.show()

# covar_matrix = PCA(n_components = 66)
# covar_matrix.fit(X_train_psig_n)
# plt.ylabel('Eigenvalues')
# plt.xlabel('# of Features')
# plt.title('PCA Eigenvalues')
# plt.ylim(0,max(covar_matrix.explained_variance_))
# plt.style.context('seaborn-whitegrid')
# plt.axhline(y=1, color='r', linestyle='--')
# plt.plot(covar_matrix.explained_variance_)
# plt.show()

# X_variance = covar_matrix.explained_variance_ratio_
# var = np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
# plt.ylabel('% Variance Explained')
# plt.xlabel('# of Features')
# plt.title('PCA Variance Explained')
# plt.ylim(min(var), 100.5)
# plt.style.context('seaborn-whitegrid')
# plt.axhline(y=80, color='r', linestyle='--')
# plt.plot(var)
# plt.show()

# print(var[5:12])
# print(f"From index 15 the PCA var becomes higher than 80%, namely: {var[9]}")

