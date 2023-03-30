import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from load_sets import load_D_set
from self_made import return_X_y, removal_zero_var

D_s = load_D_set()
X_d, y_d = return_X_y(D_s)

print(X_d)

def normalize_column(column):
    feat = column.to_numpy()
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    feat_scaled = minmax_scale.fit_transform(feat.reshape(-1, 1))
    return pd.Series(feat_scaled.flatten(), index=column.index)

X_d_normalized = X_d.apply(normalize_column)
print(X_d_normalized)
