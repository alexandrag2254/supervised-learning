"""
	Import all necessary libraries
"""

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

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)


cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)

****************************

print("Accuracy on the training subset: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on the test subset: {:.3f}".format(tree.score(X_test, y_test)))

from sklearn.ensemble import GradientBoostingClassifier as gbm
original_params = {'n_estimators': 50, 'random_state': 2}

plt.figure()
for label, color, setting in [('Depth 2, lr = 1.0', 'turquoise', {'learning_rate': 1.0, 'max_depth': 2}),
                              ('Depth 4, lr = 1.0', 'cadetblue',      {'learning_rate': 1.0, 'max_depth': 4}),
                              ('Depth 6, lr = 1.0', 'blue',      {'learning_rate': 1.0, 'max_depth': 6}),
                              ('Depth 2, lr = 0.1', 'orange',    {'learning_rate': 0.1, 'max_depth': 2}),
                              ('Depth 4, lr = 0.1', 'red',    {'learning_rate': 0.1, 'max_depth': 6}),
                              ('Depth 6, lr = 0.1', 'purple',      {'learning_rate': 0.1, 'max_depth': 6})]:
    params = dict(original_params)
    params.update(setting)
    clf = gbm(**params)
    clf.fit(X_train, y_train)

    # compute test set auc
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_predict_proba(X_test)):
        test_deviance[i] = metrics.roc_auc_score(y_test, y_pred[:,1])
    #print test auc
    plt.plot((np.arange(test_deviance.shape[0]) + 1), test_deviance,
            '-', color=color, label=label)

plt.legend(loc='lower right')
pyplot.ylim(0.90, 1.0)
plt.xlabel('Boosting Iterations')
pyplot.ylabel("validation auc")
plt.figure(figsize=(12,12))
plt.show()

"""
	The resulting graph is the boostingiterationsvsAUC.png in this folder structure

	
"""