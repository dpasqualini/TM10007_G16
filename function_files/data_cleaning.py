import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.feature_selection import VarianceThreshold


from self_made import return_X_y, removal_zero_var, load_D_set, normalize_column

D_s = load_D_set()
X_d, y_d = return_X_y(D_s)

print(X_d)


def removal_zero_var(feature_df):
    '''
    This function will remove zero-variance features from the dataset
    '''
    filter = VarianceThreshold(threshold=0.0)
    df_filter = filter.fit_transform(feature_df)
    features_kept = filter.get_support()
    names = feature_df.columns[features_kept]
    df_filter = pd.DataFrame(df_filter, columns=names)

    return df_filter

X_d_remove_zero = removal_zero_var(X_d)
print(X_d_remove_zero)

def removal_zeros(feature_df):
    zero_percents = (feature_df == 0).sum() / len(feature_df) * 100
    threshold = 50
    nonzero_cols = feature_df.columns[zero_percents < threshold]
    zero_cols = feature_df.columns[zero_percents > threshold]
    feature_df = feature_df.loc[:, nonzero_cols]
    return feature_df, zero_cols

X_d_remove_zeroperc, zero_cols = removal_zeros(X_d)
print(f"amount of columns containing {len(zero_cols)} and there called {zero_cols}")
print(X_d.loc[:, zero_cols[0]])
print(X_d_remove_zeroperc)

# def removal_NaNs(feature_df):
#     zero_percents = (feature_df == NaN).sum() / len(feature_df) * 100
#     threshold = 50
#     nonzero_cols = feature_df.columns[zero_percents < threshold]
#     zero_cols = feature_df.columns[zero_percents > threshold]
#     feature_df = feature_df.loc[:, nonzero_cols]
#     return feature_df, zero_cols

# X_d_remove_zeroperc, zero_cols = removal_zeros(X_d)
# print(f"amount of columns containing {len(zero_cols)} and there called {zero_cols}")
# print(X_d_remove_zeroperc)