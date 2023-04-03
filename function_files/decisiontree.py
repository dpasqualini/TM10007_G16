import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from self_made import load_Tr_set, load_Va_set, normalize_column
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

X_train, y_train = load_Tr_set(6)
y_train = y_train.to_numpy()
X_val, y_val = load_Va_set(6)
y_val = y_val.to_numpy()

X_train_n = X_train.apply(normalize_column)
X_val_n = X_val.apply(normalize_column)

X_train_n = X_train_n.to_numpy()
X_val_n = X_val_n.to_numpy()

clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=5)

scores = cross_val_score(clf, X_train_n, y_train, cv=10, scoring='f1_macro')

# Train the classifier on the training data
print("Cross-validation scores: ", scores)

clf.fit(X_train_n, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_val_n)

# Evaluate the performance of the classifier
f1 = f1_score(y_val, y_pred, average='macro')
print("F1-score: ", f1)
