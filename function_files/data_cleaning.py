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
print(f"amount of columns containing zero are {len(zero_cols)} and there called {zero_cols}")
print(X_d[zero_cols[4]])
print(X_d_remove_zeroperc)

def removal_nans(feature_df):
    nan_percents = feature_df.isna().sum() / len(feature_df) * 100
    threshold_nan = 50
    nonnan_cols = feature_df.columns[nan_percents < threshold_nan]
    nan_cols = feature_df.columns[nan_percents > threshold_nan]
    feature_df = feature_df.loc[:, nonnan_cols]
    return feature_df, nan_cols

# checking whether the NaNs are found like this
# X_d_nan = X_d.copy()
# X_d_nan['PREDICT_original_sf_compactness_avg_2.5D'] = np.nan
X_d_remove_nanperc, nan_cols = removal_nans(X_d)
print(f"amount of columns containing NaN are {len(nan_cols)} and there called {nan_cols}")
print(X_d_remove_nanperc)

def preprocessing(feature_df):
    '''
    This function preprocesses the features by removing zero-variance, zero-filled, and nan-filled features from the dataset.
    '''
    filter = VarianceThreshold(threshold=0.0)
    df_filter = filter.fit_transform(feature_df)
    features_kept = filter.get_support()
    names = feature_df.columns[features_kept]
    #df_filter = pd.DataFrame(df_filter, columns=names)

    # remove zero-filled features (if >50% is zero)
    zero_percents = (feature_df == 0).sum() / len(feature_df) * 100
    threshold = 50
    nonzero_cols = feature_df.columns[zero_percents < threshold]
    zero_cols = feature_df.columns[zero_percents > threshold]

    # remove nan-filled features (if >50% is NaN)
    nan_percents = feature_df.isna().sum() / len(feature_df) * 100
    threshold_nan = 50
    nonnan_cols = feature_df.columns[nan_percents < threshold_nan]
    nan_cols = feature_df.columns[nan_percents > threshold_nan]

    common_cols = set(nonnan_cols).intersection(set(nonzero_cols)).intersection(set(names))
    feature_df_chang = feature_df[list(common_cols)]


    return feature_df_chang, nan_cols, zero_cols, names

X_d_preprocessed, nan_cols, zero_cols, names = preprocessing(X_d)
print(f"The new preprocessed dataset is {X_d_preprocessed}")
print(f"The columns with >50% NaN are {nan_cols}")
print(f"The columns with >50% zero are {zero_cols}")
print(f"The columns with zero variance are {names}")
