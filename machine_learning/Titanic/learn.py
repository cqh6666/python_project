# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     learn
   Description :
   Author :       cqh
   date：          2021/4/26 16:41
-------------------------------------------------
   Change Activity:
                   2021/4/26:
-------------------------------------------------
"""
__author__ = 'cqh'

#%%
import pandas as pd
import os

#%%

run_path = 'D:\python_project\machine_learning\Titanic'
train_data_path = os.path.join(run_path,'titanic/train.csv')

#%%


train_df = pd.read_csv(train_data_path)
train_df.columns

#%%