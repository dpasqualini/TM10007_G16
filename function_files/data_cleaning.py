import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.feature_selection import VarianceThreshold


from self_made import return_X_y, load_D_set, normalize_column

D_s = load_D_set()
X_d, y_d = return_X_y(D_s)

# # print(X_d)


# def removal_zero_var(feature_df):
#     '''
#     This function will remove zero-variance features from the dataset
#     '''
#     filter = VarianceThreshold(threshold=0.0)
#     df_filter = filter.fit_transform(feature_df)
#     features_kept = filter.get_support()
#     names = feature_df.columns[features_kept]
#     df_filter = pd.DataFrame(df_filter, columns=names)

#     return df_filter


# X_d_remove_zero = removal_zero_var(X_d)
# # print(X_d_remove_zero)


# def removal_zeros(feature_df):
#     zero_percents = (feature_df == 0).sum() / len(feature_df) * 100
#     threshold = 50
#     nonzero_cols = feature_df.columns[zero_percents < threshold]
#     zero_cols = feature_df.columns[zero_percents > threshold]
#     feature_df = feature_df.loc[:, nonzero_cols]
#     return feature_df, zero_cols


# X_d_remove_zeroperc, zero_cols = removal_zeros(X_d)
# # print(f"amount of columns containing zero are {len(zero_cols)} and there called {zero_cols}")
# # print(X_d[zero_cols[4]])
# # print(X_d_remove_zeroperc)


# def removal_nans(feature_df):
#     nan_percents = feature_df.isna().sum() / len(feature_df) * 100
#     threshold_nan = 50
#     nonnan_cols = feature_df.columns[nan_percents < threshold_nan]
#     nan_cols = feature_df.columns[nan_percents > threshold_nan]
#     feature_df = feature_df.loc[:, nonnan_cols]
#     return feature_df, nan_cols


# # checking whether the NaNs are found like this
# # X_d_nan = X_d.copy()
# # X_d_nan['PREDICT_original_sf_compactness_avg_2.5D'] = np.nan
# X_d_remove_nanperc, nan_cols = removal_nans(X_d)
# # print(f"amount of columns containing NaN are {len(nan_cols)} and there called {nan_cols}")
# # print(X_d_remove_nanperc)


def preprocessing(feature_df):
    '''
    This function preprocesses the features by removing zero-variance, zero-filled,
    and nan-filled features from the dataset.
    '''

    filter = VarianceThreshold(threshold=0.0)
    df_filter1 = filter.fit_transform(feature_df)
    features_kept = filter.get_support()
    names = feature_df.columns[features_kept]
    df_filter2 = pd.DataFrame(df_filter1, columns=names)
    zero_var = pd.DataFrame(feature_df.columns[~features_kept], columns=["removed_features"])

    # remove zero-filled features (if >50% is zero)
    zero_percents = (df_filter2 == 0).sum() / len(df_filter2) * 100
    threshold = 80
    nonzero_cols = df_filter2.columns[zero_percents < threshold]
    zero_cols = df_filter2.columns[zero_percents > threshold]
    df_filter3 = pd.DataFrame(df_filter2, columns=nonzero_cols)

    # remove nan-filled features (if >50% is NaN)
    nan_percents = df_filter3.isna().sum() / len(df_filter3) * 100
    threshold_nan = 50
    nonnan_cols = df_filter3.columns[nan_percents < threshold_nan]
    nan_cols = df_filter3.columns[nan_percents > threshold_nan]
    df_filtered = pd.DataFrame(df_filter3, columns=nonnan_cols)

    print(f"Number of removed columns due to zero-variance: {len(zero_var)}")
    print(f"Number of removed non-zerovariance columns due to fraction zero > 50%: {len(zero_cols)}")
    print(f"Number of removed non-zerovariance columns due to fraction NaN > 50%:  {len(nan_cols)}")

    return df_filtered, zero_var, zero_cols, nan_cols


X_d_preprocessed, zero_var, zero_cols, nan_cols = preprocessing(X_d)
