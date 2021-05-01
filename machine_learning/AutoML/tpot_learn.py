# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     automl_learn
   Description :
   Author :       cqh
   date：          2021/4/29 21:33
-------------------------------------------------
   Change Activity:
                   2021/4/29:
-------------------------------------------------
"""
__author__ = 'cqh'
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)


print(tpot.score(X_test, y_test))
tpot.export('tpot_exported_pipeline.py')