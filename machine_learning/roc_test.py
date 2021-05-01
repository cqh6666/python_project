print(__doc__)

import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn import datasets,svm
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X = iris.data
y = iris.target

X, y = X[y != 2], y[y != 2] # (100,4)

# add noisy
random_state = np.random.RandomState(0)
n_sample, n_feature = X.shape
X = np.c_[X, random_state.randn(n_sample, n_feature*200)]       # (100,804)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

svm = svm.SVC(kernel='linear',probability=True,random_state=random_state)

y_score = svm.fit(X_train, y_train).decision_function(X_test)

fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

