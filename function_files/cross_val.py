import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from self_made import load_D_set, return_X_y, removal_zero_var, create_cross_val_sets

D_s = load_D_set()
X_d, y_d = return_X_y(D_s)


def load_Tr_set(n):
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
    X_Va_s = "X_Va_s" + str(n)
    y_Va_s = "y_Va_s" + str(n)

    this_directory = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(this_directory)
    data_folder = os.path.join(root, 'datasets')
    data_folder_Tr = os.path.join(data_folder, 'va_sets')
    X_Va_s = pd.read_csv(os.path.join(data_folder_Tr, 'X_Va_s{}.csv'.format(n)), index_col=0)
    y_Va_s = pd.read_csv(os.path.join(data_folder_Tr, 'y_Va_s{}.csv'.format(n)), index_col=0)

    return X_Va_s, y_Va_s


# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
