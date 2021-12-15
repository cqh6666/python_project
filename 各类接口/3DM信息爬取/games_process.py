# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     games_process
   Description :
   Author :       cqh
   date：          2021/11/28 12:45
-------------------------------------------------
   Change Activity:
                   2021/11/28:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd

MB_str = 'M'
GB_str = 'G'

game_pd = pd.read_csv("./3dm_game_list.csv")
game_pd.rename(columns={"size": "size / GB"})

for i in range(len(game_pd['size'])):
    temp_size_str = game_pd['size'][i]
    if MB_str in temp_size_str:
        real_size = temp_size_str.split(MB_str)[0].strip()
        game_pd['size'][i] = float(real_size) / 1024
    elif GB_str in temp_size_str:
        real_size = temp_size_str.split(GB_str)[0].strip()
        game_pd['size'][i] = float(real_size)

game_pd.to_csv('3dm_game_list_process.csv', mode='w', encoding="utf-8-sig", index=False)
