import pandas as pd
import os
import numpy as np
import matplotlib as plt

from load_sets import load_D_set
from return_X_y import return_X_y

D_s = load_D_set()
X_d, y_d = return_X_y(D_s)


def removal_zero_var(feature_df):
    '''
    This function will remove zero-variance features from the dataset
    '''
    df = feature_df

    return df

