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


# # Define the normalization function
# def normalize_column(X_d):
#     for column in X_d:
#         feat = X_d[column].to_numpy()
#         minmax_scale = MinMaxScaler(feature_range=(0, 1))
#         feat_scaled = minmax_scale.fit_transform(feat.reshape(-1, 1))
#     return feat_scaled


# X_d_normalized = X_d.apply(normalize_column)
# print(X_d_normalized)

# features = []
# for column in X_d:
#     feat = X_d[column].to_numpy()
#     minmax_scale = MinMaxScaler(feature_range=(0, 1))
#     feat_scaled = minmax_scale.fit_transform(feat.reshape(-1, 1))
#     features.append(feat_scaled)

# print(features.shape)


# feat1 = X_d.iloc[:, 0].to_numpy()
# feat2 = X_d.iloc[:, 8].to_numpy()
# # print(feat1)
# # print(feat2)

# minmax_scale = MinMaxScaler(feature_range=(0, 1))

# feat1_scaled = minmax_scale.fit_transform(feat1.reshape(-1, 1))
# feat2_scaled = minmax_scale.fit_transform(feat2.reshape(-1, 1))

#  define min max scaler
# scaler1 = MinMaxScaler()
# scaler1.fit(feat1)

# scaler2 = MinMaxScaler()
# scaler2.fit(feat2)

# # To scale data
# feat1_scaled = scaler1.transform(feat1.values.reshape(-1,1))
# feat2_scaled = scaler2.transform(feat2.values.reshape(-1,1))

# print(f"feature 2", feat2)
# print(f"feat2 scaled", feat2_scaled)

# fig, (ax1, ax2) = plt.subplots(2)
# fig.suptitle('Scaling data try 1')
# ax1.scatter(feat1, feat2)
# ax1.set_title('Original data')
# ax2.scatter(feat1_scaled, feat2_scaled)
# ax2.set_title('Scaled data')
# plt.show()
