# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     TikTok_download
   Description :
   Author :       cqh
   date：          2021/4/30 17:18
-------------------------------------------------
   Change Activity:
                   2021/4/30:
-------------------------------------------------
"""
__author__ = 'cqh'

import requests

url = "https://api.tx7.co/api/NetEaseCloud"
payload = {
    'id': "1827600686"
}
res = requests.get(url, params=payload)
print(res.json())
