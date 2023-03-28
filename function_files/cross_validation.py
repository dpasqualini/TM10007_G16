import pandas as pd
import os
import numpy as np
import matplotlib as plt

from sklearn.model_selection import train_test_split


from load_sets import load_D_set
from self_made import return_X_y, removal_zero_var

D_s = load_D_set()
X_d, y_d = return_X_y(D_s)


def cross_validation(data_set):
    '''
    This function will perform a stratified split of the data, and will result in 10 pairs of train and validation sets,
    consisting out of 90% and 10% of the design set, respectively.
    '''

    for i in range(1, 8):
        X_Va_s = "X_Vs_s" + str(i)
        X_Tr_s = "X_Tr_s" + str(i)

        if i == 1:
            X_Va_s = data_set.iloc[:(i*int(0.09*len(data_set))+i)]
            X_Tr_s = data_set.iloc[i*int(0.09*len(data_set))+i:]  # Deze term moet nu verder uitgewerkt worden!
        elif 1 < i and i <= 5:
            X_Va_s = data_set.iloc[((i-1)*int(0.09*len(data_set))+(i-1)):(i*int(0.09*len(data_set))+i)]
            X_Tr_s = data_set.iloc[i*int(0.09*len(data_set)):]  # Deze term moet nu verder uitgewerkt worden!
        else:
            X_Va_s = data_set.iloc[5+(i-1)*int(0.09*len(data_set)):5+i*int(0.09*len(data_set))]
            X_Tr_s = data_set.iloc[i*int(0.09*len(data_set)):]  # Deze term moet nu verder uitgewerkt worden!

        this_directory = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(this_directory)
        data_folder = os.path.join(root, 'datasets')
        X_Tr_s.to_csv(os.path.join(data_folder, 'X_Tr_s{}.csv'.format(i)))
        X_Va_s.to_csv(os.path.join(data_folder, 'X_Va_s{}.csv'.format(i)))

        if i == 10:
            print(f'{i*2} Files have been created')


cross_validation(D_s)
