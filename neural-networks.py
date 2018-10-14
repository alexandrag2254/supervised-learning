"""
	Neural Networks
"""

# Import all necessary libraries
import os
import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns 
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)

print("Accuracy on the training subset: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on the test subset: {:.3f}".format(tree.score(X_test, y_test)))

