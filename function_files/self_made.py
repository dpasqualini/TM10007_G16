import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics


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

    # remove features with standard deviation lower than 0.1
    std_dev = df_filter2.std()
    low_std_dev = std_dev[std_dev < 0.1].index.tolist()
    df_filter3 = df_filter2.drop(columns=low_std_dev)

    # remove zero-filled features (if >50% is zero)
    zero_percents = (df_filter3 == 0).sum() / len(df_filter3) * 100
    threshold = 80
    nonzero_cols = df_filter3.columns[zero_percents <= threshold]
    zero_cols = df_filter3.columns[zero_percents > threshold]
    df_filter4 = pd.DataFrame(df_filter3, columns=nonzero_cols)

    # remove nan-filled features (if >50% is NaN)
    nan_percents = df_filter4.isna().sum() / len(df_filter3) * 100
    threshold_nan = 50
    nonnan_cols = df_filter4.columns[nan_percents <= threshold_nan]
    nan_cols = df_filter4.columns[nan_percents > threshold_nan]
    df_filtered = pd.DataFrame(df_filter4, columns=nonnan_cols)

    print(f"Number of removed columns due to zero-variance: {len(zero_var)}")
    print(f"Number of removed columns due to standard deviation < 0.1: {len(low_std_dev)}")
    print(f"Number of removed non-zero-variance columns due to fraction zero > 80%: {len(zero_cols)}")
    print(f"Number of removed non-zero-variance columns due to fraction NaN > 50%:  {len(nan_cols)}")
    print(f'Remaining number of features after preprocessing: {df_filtered.shape[1]}')

    return df_filtered, zero_var, low_std_dev, zero_cols, nan_cols


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


# def normalize_column2(column2):
#     '''
#     This function will normalize a panda Series using the standard scaling.
#     It will return a new column with normalized numbers with a mean of 0 and a standard deviation of 1. 
#     Assumed is that the data has a normal distribution. To apply this to a dataframe
#     this function needs to be applied to each column (which can be done with the use of .apply())
#     '''
#     feat2 = column2.to_numpy()
#     standard_scale2 = StandardScaler()
#     feat_scaled2 = standard_scale2.fit_transform(feat2.reshape(-1, 1))
#     return pd.Series(feat_scaled2.flatten(), index=column2.index)


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


def PCA_info_notransform(X, selected_features):
    # Covariance matrix
    covar_matrix = PCA(n_components = len(selected_features))
    covar_matrix.fit(X)
    eigenvalues = covar_matrix.explained_variance_

    variance = covar_matrix.explained_variance_ratio_
    var = np.cumsum(np.round(variance, decimals=3)*100)

    return variance, var

def PCA_plot(var):
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Variance Explained')
    plt.ylim(min(var), 100.5)
    plt.style.context('seaborn-whitegrid')
    plt.axhline(y=90, color='r', linestyle='--')
    plt.plot(var)
    plt.show()
    
    for j, val in enumerate(var):
        if val > 90:
            print("The index of the first number that is bigger than 90% is:", j)
            n_comp = j+1
            break
    return n_comp

def PCA_transform(X_tr_normalized, n_comp):
    pca = PCA(n_components=n_comp) # as index starts at 0 the number of components is one higher
    X_pca = pca.fit_transform(X_tr_normalized)

    eigenvalues = pca.explained_variance_

    variance = pca.explained_variance_ratio_
    var = np.cumsum(np.round(variance, decimals=3)*100)

    return pca, X_pca, variance, var

def PCA_transform_overview(variance, var):
    n_components = len(variance)
    idx = np.arange(n_components)+1

    df_explained_variance = pd.DataFrame([variance, var], 
                                        index=['explained variance', 'cumulative [%]'], 
                                        columns=idx).T

    mean_explained_variance = df_explained_variance.iloc[:,0].mean() # calculate mean explained variance

    # DISPLAY info about PCs
    print('PCA Overview')
    print('='*40)
    print("Total: {} components".format(n_components))
    print('-'*40)
    print('Mean explained variance:', round(mean_explained_variance,3))
    print('-'*40)
    print(df_explained_variance)
    print('-'*40)
    return df_explained_variance, mean_explained_variance, idx

def scree_plot(df_explained_variance, mean_explained_variance, idx):
    limit_df = 15 # change this number if you want to have more or less PCA components visualized
    df_explained_variance_limited = df_explained_variance.iloc[:limit_df,:]

    #make scree plot
    fig, ax1 = plt.subplots(figsize=(15,6))

    ax1.set_title('Explained variance across principal components', fontsize=14)
    ax1.set_xlabel('Principal component', fontsize=12)
    ax1.set_ylabel('Explained variance', fontsize=12)

    ax2 = sns.barplot(x=idx[:limit_df], y='explained variance', data=df_explained_variance_limited, palette='summer')
    ax2 = ax1.twinx()
    ax2.grid(False)

    ax2.set_ylabel('Cumulative', fontsize=14)
    ax2 = sns.lineplot(x=idx[:limit_df]-1, y='cumulative [%]', data=df_explained_variance_limited, color='#fc8d59')

    ax1.axhline(mean_explained_variance, ls='--', color='#fc8d59') #plot mean
    ax1.text(-.8, mean_explained_variance+(mean_explained_variance*.05), "average", color='#fc8d59', fontsize=14) #label y axis

    max_y1 = max(df_explained_variance_limited.iloc[:,0])
    max_y2 = max(df_explained_variance_limited.iloc[:,1])
    ax1.set(ylim=(0, max_y1+max_y1*.1))
    ax2.set(ylim=(0, max_y2+max_y2*.1))
    
    return plt.show()

def factor_loadings(X, pca, df_explained_variance):
    ############### DEFINE PARAMETERS 
    selected_features = X.columns
    top_k = 3
    #select data based on percentile (top_q) or top-k features
    top_q = .50
    top_pc = 3
    ###############

    # PCA factor loadings
    df_c = pd.DataFrame(pca.components_, columns=selected_features).T

    print("Factor Loadings for the 1. component \n(explains {0:.2f} of the variance)".format(df_explained_variance.iloc[0,0]))
    print('='*40,'\n')
    print('Top {} highest'.format(top_k))
    print('-'*40)
    print(df_c.iloc[:,0].sort_values(ascending=False)[:top_k], '\n')

    print('Top {} lowest'.format(top_k))
    print('-'*40)
    print(df_c.iloc[:,0].sort_values()[:top_k])

    # Plot heatmap
    size_yaxis = round(X[selected_features].shape[1] * 0.5)
    # Create a mask of values above 0.025
    mask = df_c > 0.025

    fig, ax = plt.subplots(figsize=(8,size_yaxis))
    sns.heatmap(df_c.iloc[:,:], annot=True, cmap="YlGnBu", ax=ax)
    return plt.show()

def classifiers(clsfs, X_pca, y_train):
    f1_scores = []
    # Loop over each classifier and fit the model, make predictions and compute metrics
    for clf in clsfs:
        clf.fit(X_pca, y_train)
        y_pred = clf.predict(X_pca)
        accuracy = metrics.accuracy_score(y_train, y_pred)
        F1 = metrics.f1_score(y_train, y_pred)
        f1_scores.append(F1)
        precision = metrics.precision_score(y_train, y_pred)
        recall = metrics.recall_score(y_train, y_pred)
        print(type(clf).__name__)
        print('Accuracy:', accuracy)
        print('F1:', F1)
        print('Precision:', precision)
        print('Recall:', recall)
        print()
    return f1_scores

