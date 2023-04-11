import pandas as pd
import os


def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Liver_radiomicFeatures.csv'), index_col=0)

    return data


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
    data_folder_Va = os.path.join(data_folder, 'va_sets')
    X_Va_s = pd.read_csv(os.path.join(data_folder_Va, 'X_Va_s{}.csv'.format(n)), index_col=0)
    y_Va_s = pd.read_csv(os.path.join(data_folder_Va, 'y_Va_s{}.csv'.format(n)), index_col=0)

    return X_Va_s, y_Va_s
