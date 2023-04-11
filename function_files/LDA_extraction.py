import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from self_made import load_Tr_set, load_Va_set, normalize_column
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def plt_lda(df_features, labels, k=10):
    '''
    Using this function will return a 1 dimensional LDA plot. May be used as feature extraction.
    '''

    labels = labels['label'].replace({'benign': 0, 'malignant': 1})
    labels = labels.to_numpy()
    ny_features = df_features.to_numpy()

    # Perform LDA
    lda = LDA()
    X_lda = lda.fit_transform(ny_features, labels)
    lda_coeffs = lda.coef_[0]

    sorted_coeffs = sorted(lda_coeffs, key=lambda x: abs(x), reverse=True)
    selected_indices = [i for i in range(len(sorted_coeffs)) if abs(lda_coeffs[i]) in sorted_coeffs[:k]]
    selected_features = df_features.columns[selected_indices].tolist()
    df_features_selected = df_features[selected_features]

    # Plot the t-SNE representation colored by the labels
    # plt.scatter(X_lda, np.zeros_like(X_lda), c=labels)
    # plt.xlabel('LD1')
    # plt.ylim(-0.1, 0.1)
    # plt.show()

    return df_features_selected


X_train, y_train = load_Tr_set(1)
X_val, y_val = load_Va_set(1)

X_train_n = X_train.apply(normalize_column)
X_val_n = X_val.apply(normalize_column)

X_selected = plt_lda(X_train_n, y_train, 30)
# print(X_selected.shape)

# svm = SVC(kernel='sigmoid', C=1.0, random_state=42)
# svm.fit(X_selected, y_train.values.ravel())

# X_val_selected = X_val_n[X_selected.columns]  # Select the same columns as in X_selected
# y_pred = svm.predict(X_val_selected)

# accuracy = accuracy_score(y_val, y_pred)
# print("Accuracy:", accuracy)

# fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))

# y_train = y_train['label'].replace({'benign': 0, 'malignant': 1})
# y_train = y_train.to_numpy()

# # Loop over each subplot and plot the selected features
# for i, ax in enumerate(axs.flat):
#     x_feature_index = i * 2
#     y_feature_index = i * 2 + 1

#     x_feature = X_selected.columns[x_feature_index]
#     y_feature = X_selected.columns[y_feature_index]
#     ax.scatter(X_selected.loc[:, x_feature], X_selected.loc[:, y_feature], c=y_train)
#     ax.set_xlabel(f"Feature {x_feature}")
#     ax.set_ylabel(f"Feature {y_feature}")
#     ax.set_title(f"Plot of features {x_feature} and {y_feature}")

# plt.tight_layout()
# plt.show()


rf = RandomForestClassifier()
y_train = y_train.to_numpy()

# Train the model using the training data and the selected features
rf.fit(X_selected, y_train)

# Predict the labels for the test data using the trained model
y_pred = rf.predict(X_val_n[X_selected.columns])

scores = cross_val_score(rf, X_train_n, y_train, cv=10, scoring='f1_macro')

# Evaluate the performance of the model using accuracy
f1 = f1_score(y_val, y_pred, average='macro')
print("F1-score: ", f1)
