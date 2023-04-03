import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from self_made import load_Tr_set, load_Va_set, normalize_column
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


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
    plt.scatter(X_lda, np.zeros_like(X_lda), c=labels)
    plt.xlabel('LD1')
    plt.ylim(-0.1, 0.1)
    plt.show()

    return df_features_selected


X_train, y_train = load_Tr_set(1)
X_val, y_val = load_Va_set(1)

X_train_n = X_train.apply(normalize_column)
X_val_n = X_val.apply(normalize_column)

X_selected = plt_lda(X_train_n, y_train)

print(X_selected.describe(include='all'))
