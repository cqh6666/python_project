# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     LIBSVM_test
   Description :
   Author :       cqh
   date：          2021/4/19
-------------------------------------------------
   Change Activity:
                   2021/4/19:
-------------------------------------------------
"""
__author__ = 'cqh'

# %%

from libsvm.svmutil import *
from libsvm.svm import *
from sklearn import svm
from sklearn.metrics import accuracy_score

#%%

y, x = svm_read_problem('./train.txt')
yt, xt = svm_read_problem('./test.txt')
y, x
# %%
problem = svm_problem(y, x)
model = svm_train(problem)
p_label, p_acc, p_val = svm_predict(yt, xt, model)
svm_save_model('svm_model', model)

# %%



# 转化为sklearn能用的数据库

TrainingSet = []
TestingSet = []

for i in x:
    TrainingSet.append(list(i.values()))

for j in xt:
    TestingSet.append(list(j.values()))

clf = svm.SVC(kernel='rbf')
clf.fit(TrainingSet, y)

predict_y = clf.predict(TestingSet)
accuracy_score(predict_y, yt)
