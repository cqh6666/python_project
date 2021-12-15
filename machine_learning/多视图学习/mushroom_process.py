# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     mushroom_process
   Description :
   Author :       cqh
   date：          2021/11/17 10:33
-------------------------------------------------
   Change Activity:
                   2021/11/17:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd
file_path ='./dataset/mushroom.csv'
names = ['class','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']

mushroom = pd.read_csv(file_path,header=None,names=names)

# 展示详情信息
# print(mushroom.info())
# print(mushroom.describe())
