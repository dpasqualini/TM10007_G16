import pandas as pd
import os
import numpy as np
import matplotlib as plt

from load_sets import load_D_set
from self_made import return_X_y, removal_zero_var

D_s = load_D_set()
X_d, y_d = return_X_y(D_s)

print(X_d.head())
