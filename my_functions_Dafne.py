''' Importing data'''

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import VarianceThreshold


def load_Ft_set():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(this_directory)
    data_folder = os.path.join(root, 'datasets')
    Ft_usable = pd.read_csv(os.path.join(data_folder, 'Ft_set.csv'), index_col=0)

    return Ft_usable


def load_D_set():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(this_directory)
    data_folder = os.path.join(root, 'datasets')
    D_usable = pd.read_csv(os.path.join(data_folder, 'D_set.csv'), index_col=0)

    return D_usable


def return_X_y(df):
    '''
    Returns dataframe y_d including the labels, and a seperate dataframe X_d including the feature information.
    '''
    y_d = df.pop('label').to_frame()
    X_d = df
    return X_d, y_d


def second_split(data_set, labels):
    '''
    This function will perform a stratified split of the data, and will result in 10 pairs of train and validation sets,
    consisting out of 90% and 10% of the design set, respectively.
    '''

    for i in range(1, 12):
        X_Va_s = "X_Vs_s" + str(i)
        X_Tr_s = "X_Tr_s" + str(i)
        X_Tr_s2 = "X_Tr_s2" + str(i)

        y_Va_s = "y_Vs_s" + str(i)
        y_Tr_s = "y_Tr_s" + str(i)
        y_Tr_s2 = "y_Tr_s2" + str(i)

        if i == 1:
            X_Va_s = data_set.iloc[: (i*int(0.09*len(data_set))+i)]
            X_Tr_s = data_set.iloc[(i*int(0.09*len(data_set))+i):]

            y_Va_s = labels.iloc[: (i*int(0.09*len(labels))+i)]
            y_Tr_s = labels.iloc[(i*int(0.09*len(labels))+i):]

        elif 1 < i and i <= 5:
            X_Va_s = data_set.iloc[((i-1)*int(0.09*len(data_set))+(i-1)): (i*int(0.09*len(data_set))+i)]
            X_Tr_s = data_set.iloc[: ((i-1)*int(0.09*len(data_set))+(i-1))]
            X_Tr_s2 = data_set.iloc[(i*int(0.09*len(data_set))+i):]
            X_Tr_s = pd.concat([X_Tr_s, X_Tr_s2], ignore_index=False)

            y_Va_s = labels.iloc[((i-1)*int(0.09*len(labels))+(i-1)): (i*int(0.09*len(labels))+i)]
            y_Tr_s = labels.iloc[: ((i-1)*int(0.09*len(labels))+(i-1))]
            y_Tr_s2 = labels.iloc[(i*int(0.09*len(labels))+i):]
            y_Tr_s = pd.concat([y_Tr_s, y_Tr_s2], ignore_index=False)

        else:
            X_Va_s = data_set.iloc[((i-1)*int(0.09*len(data_set))+5): (i*int(0.09*len(data_set))+5)]
            X_Tr_s = data_set.iloc[: (((i-1)*int(0.09*len(data_set))+5))]
            X_Tr_s2 = data_set.iloc[(i*int(0.09*len(data_set))+5):]
            X_Tr_s = pd.concat([X_Tr_s, X_Tr_s2], ignore_index=False)

            y_Va_s = labels.iloc[((i-1)*int(0.09*len(labels))+5): (i*int(0.09*len(labels))+5)]
            y_Tr_s = labels.iloc[: (((i-1)*int(0.09*len(labels))+5))]
            y_Tr_s2 = labels.iloc[(i*int(0.09*len(labels))+5):]
            y_Tr_s = pd.concat([y_Tr_s, y_Tr_s2], ignore_index=False)

        this_directory = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(this_directory)
        data_folder = os.path.join(root, 'datasets')
        data_folder_Tr = os.path.join(data_folder, 'tr_sets')
        data_folder_Va = os.path.join(data_folder, 'va_sets')
        X_Tr_s.to_csv(os.path.join(data_folder_Tr, 'X_Tr_s{}.csv'.format(i)))
        X_Va_s.to_csv(os.path.join(data_folder_Va, 'X_Va_s{}.csv'.format(i)))
        y_Tr_s.to_csv(os.path.join(data_folder_Tr, 'y_Tr_s{}.csv'.format(i)))
        y_Va_s.to_csv(os.path.join(data_folder_Va, 'y_Va_s{}.csv'.format(i)))

        if i == 11:
            print(f'{i*2*2} Files have been created, in folder "datasets"')


def load_Tr_set(n):
    '''
    This function will load two dataframes, first containing the feature information and the second one containing
    the labels from the specified training set.
    '''

    X_Tr_s = "X_Tr_s" + str(n)
    y_Tr_s = "y_Tr_s" + str(n)

    this_directory = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(this_directory)
    data_folder = os.path.join(root, 'datasets')
    data_folder_Tr = os.path.join(data_folder, 'tr_sets')
    X_Tr_s = pd.read_csv(os.path.join(data_folder_Tr, 'X_Tr_s{}.csv'.format(n)), index_col=0)
    y_Tr_s = pd.read_csv(os.path.join(data_folder_Tr, 'y_Tr_s{}.csv'.format(n)), index_col=0)

    return X_Tr_s, y_Tr_s


def load_Va_set(n):
    '''
    This function will load two dataframes, first containing the feature information and the second one containing
    the labels from the specified validation set.
    '''
    X_Va_s = "X_Va_s" + str(n)
    y_Va_s = "y_Va_s" + str(n)

    this_directory = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(this_directory)
    data_folder = os.path.join(root, 'datasets')
    data_folder_Tr = os.path.join(data_folder, 'va_sets')
    X_Va_s = pd.read_csv(os.path.join(data_folder_Tr, 'X_Va_s{}.csv'.format(n)), index_col=0)
    y_Va_s = pd.read_csv(os.path.join(data_folder_Tr, 'y_Va_s{}.csv'.format(n)), index_col=0)

    return X_Va_s, y_Va_s


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

    # remove zero-filled features (if >80% is zero)
    zero_percents = (df_filter2 == 0).sum() / len(df_filter2) * 100
    threshold = 80
    nonzero_cols = df_filter2.columns[zero_percents <= threshold]
    zero_cols = df_filter2.columns[zero_percents > threshold]
    df_filtered = pd.DataFrame(df_filter2, columns=nonzero_cols)

    # # remove nan-filled features (if >50% is NaN)
    # nan_percents = df_filter3.isna().sum() / len(df_filter3) * 100
    # threshold_nan = 50
    # nonnan_cols = df_filter3.columns[nan_percents <= threshold_nan]
    # nan_cols = df_filter3.columns[nan_percents > threshold_nan]
    # df_filtered = pd.DataFrame(df_filter3, columns=nonnan_cols)

    print(f"Number of removed columns due to zero-variance: {len(zero_var)}")
    print(f"Number of removed non-zero-variance columns due to fraction zero > 80%: {len(zero_cols)}")
    # print(f"Number of removed non-zero-variance columns due to fraction NaN > 50%:  {len(nan_cols)}")
    print(f'Remaining number of features after preprocessing: {df_filtered.shape[1]}')

    return df_filtered, zero_var, zero_cols


def normalize_column(column):
    '''
    This function will normalize a panda Series using the minimum-maximum scaling.
    It will return a new column with normalized numbers between 0 and 1. To apply this to a dataframe
    this function needs to be applied to each column (which can be done with the use of .apply())
    '''
    feat = column.to_numpy()
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    feat_scaled = minmax_scale.fit_transform(feat.reshape(-1, 1))
    return pd.Series(feat_scaled.flatten(), index=column.index)


def normalize_standard(column):
    '''
    This function will normalize a panda Series using the standard scaling.
    It will return a new column with normalized numbers with a mean of 0 and a standard deviation of 1. 
    Assumed is that the data has a normal distribution. To apply this to a dataframe
    this function needs to be applied to each column (which can be done with the use of .apply())
    '''
    feat2 = column.to_numpy()
    standard_scale = StandardScaler()
    feat_scaled2 = standard_scale.fit_transform(feat2.reshape(-1, 1))
    return pd.Series(feat_scaled2.flatten(), index=column.index)


def plt_tsne(df_features, labels):
    '''
    Using this function will return a 2 dimensional t-SNE plot.
    May be used as visualisation and possible feature extraction.
    '''

    labels = labels['label'].replace({'benign': 0, 'malignant': 1})
    labels = labels.to_numpy()
    df_features = df_features.to_numpy()

    # Perform TSNE
    tsne = TSNE(n_components=2, learning_rate="auto", perplexity=5)
    X_tsne = tsne.fit_transform(df_features, labels)

    # Plot the t-SNE representation colored by the labels
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels)
    return plt.show()


def plt_lda(df_features, labels):
    '''
    Using this function will return a 1 dimensional LDA plot. May be used as feature extraction.
    '''

    labels = labels['label'].replace({'benign': 0, 'malignant': 1})
    labels = labels.to_numpy()
    df_features = df_features.to_numpy()

    # Perform LDA
    lda = LDA()
    X_lda = lda.fit_transform(df_features, labels)

    # Plot the t-SNE representation colored by the labels
    plt.scatter(X_lda, np.zeros_like(X_lda), c=labels)
    plt.xlabel('LD1')
    plt.ylim(-0.1, 0.1)

    return plt.show()
