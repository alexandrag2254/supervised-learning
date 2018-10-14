# crediting dataschool.io tutorial "Machine learning in Python with scikit-learn" for which this code is influenced by

"""
  k-nearest neighbors (KNN)
"""

# Import all necessary libraries
import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
%matplotlib inline

# breast cancer data 
cancer = load_breast_cancer()

# plot graph of countplot number of each target in whole dataset
sns.countplot(x="target", data=cancer)

knn = KNeighborsClassifier(n_neighbors=1)
# knn = KNeighborsClassifier(n_neighbors=15)
print knn

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.4, random_state=0)

print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)

k_range = range(1, 50)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn.predict(X_test)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print scores

plt.plot(k_range,scores)

# evaluating model's predictions against the test dataset
print(metrics.classification_report(y_test, y_pred))

"""
  RESPONSE:
     precision    recall  f1-score   support

            0       0.97      0.93      0.95        83
            1       0.96      0.99      0.97       145

  avg / total       0.97      0.96      0.96       228

  Analysis:
  Of all the points labeled 1, 99% of the results returned were accurate and 
  of the total datapoints, 96% were accurate 
"""

kn = range(5,35,5)
kauc_trn, kauc_tst = np.zeros(len(kn)), np.zeros(len(kn))
for i, k in zip(range(0, len(kn)), kn):
    clf1 = KNeighborsClassifier(n_neighbors=k, algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, p=2, weights='uniform')
    clf1 = clf1.fit(X_train, y_train)
    pred_tst = clf1.predict_proba(X_test)[:,1]
    pred_trn = clf1.predict_proba(X_train)[:,1]
    kauc_tst[i] = metrics.roc_auc_score(y_test, pred_tst)
    kauc_trn[i] = metrics.roc_auc_score(y_train, pred_trn)


plt.plot(kn, kauc_tst, linewidth=3, label = "KNN test AUC")
plt.plot(kn, kauc_trn, linewidth=3, label = "KNN train AUC")
#plt.grid()
plt.legend()
plt.ylim(0.9, 1.0)
plt.xlabel("k Nearest Neighbors")
plt.ylabel("validation auc")
plt.figure(figsize=(12,12))
plt.show()

# The resulting graph is the testvstrainAUC.png in this folder structure
