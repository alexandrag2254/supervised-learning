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
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)


cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)

print("Accuracy on the training subset: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on the test subset: {:.3f}".format(tree.score(X_test, y_test)))

"""
	Result:
	Accuracy on the training subset: 1.000
	Accuracy on the test subset: 0.895
"""

# c = DecisionTreeClassifier()
c = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=1,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')

c.fit(X_train, y_train)
accu_train = np.sum(c.predict(X_train) == y_train) / float (y_train.size)
accu_test = np.sum(c.predict(X_test) == y_test) / float (y_test.size)
print "Classification accuracy on train", accu_train
print "Classification accuracy on test", accu_test

"""
	Result
	Classification accuracy on train 1.0
	Classification accuracy on test 0.8991228070175439
"""

"""
	to visualize decision tree
"""

# !pip install graphviz

# import graphviz
# dot_data = tree.export_graphviz(c, out_file=None,  feature_names=cancer.feature_names,  
#     class_names=cancer.target_names,  filled=True, rounded=True, special_characters=True) 
# graph = graphviz.Source(dot_data)  
# graph.render('dtree_render',view=True)

"""
	Graphing the 
"""
depth = 30
tree_auc_trn, tree_auc_tst = np.zeros(depth), np.zeros(depth)
for i in range(1,depth):
    clf1 = DecisionTreeClassifier(max_depth=i, criterion='entropy')
    clf1 = clf1.fit(X_train, y_train)
    tree_auc_trn[i] = metrics.roc_auc_score(y_test, clf1.predict_proba(X_test)[:,1])
    tree_auc_tst[i] = metrics.roc_auc_score(y_train, clf1.predict_proba(X_train)[:,1])
    
    from matplotlib import pyplot
pyplot.plot(tree_auc_tst, linewidth=3, label = "Decision tree test AUC")
pyplot.plot(tree_auc_trn, linewidth=3, label = "Decision tree train AUC")
pyplot.legend()
pyplot.ylim(0.8, 1.1)
pyplot.xlabel("Max_depth")
pyplot.ylabel("validation auc")
plt.figure(figsize=(12,12))
pyplot.show()

# graph resulting from this code can be seen at ____

print "Best Test Score", max(tree_auc_tst)
# Best Test Score 0.9154739764426835

"""
	The resulting graph is the depthvsAUC.png in this folder structure

	From the graph we can see the optimal depth. 
	Any deeper and the test error rate increases while the training error continues to 
	increase in what we call overfitting to the data.
"""
