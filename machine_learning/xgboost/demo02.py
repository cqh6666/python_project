# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     get_data
   Description :
   Author :       cqh
   date：          2021/9/23 20:58
-------------------------------------------------
   Change Activity:
                   2021/9/23:
-------------------------------------------------
"""
__author__ = 'cqh'

#xgboost 分类
from sklearn.datasets import load_boston
from xgboost import XGBRegressor as XGBR
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# read in the iris data
boston = load_boston()

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

reg = XGBR(n_estimators=100).fit(X_train,y_train)
y_predict = reg.predict(X_test)
score = reg.score(X_test,y_test)
mse = MSE(y_test,y_predict)
print(reg.feature_importances_)