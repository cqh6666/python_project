# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_data
   Description :
   Author :       cqh
   date：          2021/11/10 11:13
-------------------------------------------------
   Change Activity:
                   2021/11/10:
-------------------------------------------------
"""
__author__ = 'cqh'

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

breast_cancer = load_breast_cancer()

data = breast_cancer.data
labels = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(data, labels)

RF = RandomForestClassifier(n_estimators=10, random_state=11)
RF.fit(X_train, y_train)
pred = RF.predict(X_test)
print(classification_report(y_test, pred))


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=200)
clf.fit(X_train,y_train)
pred2 = clf.predict(X_test)
print(classification_report(y_test,pred2))

print("==========================================")
from sklearn.ensemble import  AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("AdaBoost")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

### GaussianNB
print("==========================================")
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("GaussianNB")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))