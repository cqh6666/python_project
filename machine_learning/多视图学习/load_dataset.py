# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     load_dataset
   Description :
   Author :       cqh
   date：          2021/11/16 19:40
-------------------------------------------------
   Change Activity:
                   2021/11/16:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd


data_set = pd.read_csv('./dataset/mushroom_data.csv')
data = data_set.values[:,:]

print(data)