# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     games_craw
   Description :
   Author :       cqh
   date：          2021/11/26 18:04
-------------------------------------------------
   Change Activity:
                   2021/11/26:
-------------------------------------------------
"""
__author__ = 'cqh'

from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import time
import random

page_size = 1104
game_pd = pd.DataFrame(
    columns=('title', 'type', 'hot', 'language', 'score', 'publish', 'platform', 'size', 'img'))
game_pd.to_csv('3dm_game_list.csv', mode='w', encoding="utf-8-sig",index=False)

for page_index in range(page_size):
    url = "https://dl.3dmgame.com/all_all_{}_time_free/".format(page_index+1)
    res = requests.get(url).text
    soup = BeautifulSoup(res, 'html.parser')
    game_text_list = soup.find('ul', attrs={'class': 'downllis'}).find_all("div", attrs={'class': 'item'})

    for game in game_text_list:
        main_content = game.find_all('li')
        game_size_str = game.find('a', attrs={'class': 'a_click'}).text
        game_size = re.findall(r'[(](.*?)[)]', game_size_str)

        game_dict = {
            'title': game.find('div', attrs={'class': 'bt'}).text,
            'type': main_content[0].contents[1].text,
            'hot': main_content[1].text,
            'language': main_content[2].contents[1].text,
            'score': main_content[3].contents[1].text,
            'publish': main_content[4].contents[1].text,
            'platform': main_content[5].contents[1].text,
            'size': game_size[0] if game_size else 0,
            'img': game.find('img')['data-original']
        }
        game_pd = game_pd.append(game_dict, ignore_index=True)

    print("=============第[{}]页共{}条数据保存成功===============".format(page_index+1,len(game_text_list)))
    if page_index % 2 == 0:
        game_pd.to_csv('3dm_game_list.csv', mode='a',encoding="utf-8-sig",header=False,index=False)
        game_pd.drop(game_pd.index, inplace=True)
        time.sleep(random.random() * 3)

print(game_pd.head())
