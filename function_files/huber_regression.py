import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from self_made import load_D_set, return_X_y, removal_zero_var, create_cross_val_sets

D_s = load_D_set()
X_d, y_d = return_X_y(D_s)

