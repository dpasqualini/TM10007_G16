import pandas as pd
import os
import numpy as np
import matplotlib as plt

from sklearn.feature_selection import VarianceThreshold


def return_X_y(df):
    '''
    Returns dataframe y_d including the labels, and a seperate dataframe X_d including the feature information.
    '''
    y_d = df.pop('label').to_frame()
    X_d = df
    return X_d, y_d


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
