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

print(y_d)
y_d_binary = y_d.copy()

# replace label 'benigne' with 0 and 'maligne' with 1
y_d_binary['label'] = y_d_binary['label'].replace({'benign': 0, 'malignant': 1})

print(y_d_binary)