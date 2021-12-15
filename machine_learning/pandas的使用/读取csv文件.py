# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     读取csv文件
   Description :
   Author :       cqh
   date：          2021/9/23 21:05
-------------------------------------------------
   Change Activity:
                   2021/9/23:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd

DATASET = "../dataset/xigua_data3.0.csv"


if __name__ == '__main__':
    data = pd.read_csv(DATASET, header=0)
    for index, row in data.iterrows():
        print(index, row)
