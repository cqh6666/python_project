# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     bili_up
   Description :
   Author :       cqh
   date：          2021/12/15 12:33
-------------------------------------------------
   Change Activity:
                   2021/12/15:
-------------------------------------------------
"""
import requests
import json
import time
import random

# 填写cookie即可运行
'''
1、浏览器登入哔哩网站
2、访问 http://api.bilibili.com/x/space/myinfo
3、F12看到cookie的值粘贴即可
'''
cookies = "buvid3=C190427C-053D-45E8-99F4-ACE9C455E52A138377infoc; rpdid=|(JYm)l~uulu0J'uY|RkukRRl; LIVE_BUVID=AUTO9716053491242908; buivd_fp=C190427C-053D-45E8-99F4-ACE9C455E52A138377infoc; buvid_fp_plain=C190427C-053D-45E8-99F4-ACE9C455E52A138377infoc; buvid_fp=C190427C-053D-45E8-99F4-ACE9C455E52A138377infoc; fingerprint3=e6422c1f47781823425eb0e53efe4691; fingerprint_s=65d08d841d0bee2dbf4fd3fb6cff6678; CURRENT_BLACKGAP=1; _uuid=A172104D-2527-2297-FA4A-841EB4C9B62927595infoc; DedeUserID=108719797; DedeUserID__ckMd5=eddcb21ea4c5e078; SESSDATA=ceaa2487%2C1651842774%2Ce69c5*b1; bili_jct=d574e035002b230f668016a669329d9b; PVID=1; video_page_version=v_old_home; fingerprint=9f55084bb2bafda7445912c682caffb6; i-wanna-go-back=1; b_ut=6; CURRENT_FNVAL=2000; blackside_state=1; CURRENT_QUALITY=80; b_lsid=7D10C7BB8_17DBC2CEE37; bp_video_offset_108719797=604304539960206225; innersign=1; sid=a0ix2zb4"
coin_operated = 0  # (投币开关，0不投币，1投币)


# cookie转字典
def extract_cookies(cookies):
    global csrf
    cookies = dict([l.split("=", 1) for l in cookies.split("; ")])
    csrf = cookies['bili_jct']
    return cookies


# 银币数
def getCoin():
    cookie = extract_cookies(cookies)
    url = "http://account.bilibili.com/site/getCoin"
    r = requests.get(url, cookies=cookie).text
    j = json.loads(r)
    money = j['data']['money']
    return int(money)


# 个人信息
def getInfo():
    global uid
    url = "http://api.bilibili.com/x/space/myinfo"
    cookie = extract_cookies(cookies)
    j = requests.get(url, cookies=cookie).json()
    username = j['data']['name']
    uid = j['data']['mid']
    level = j['data']['level']
    current_exp = j['data']['level_exp']['current_exp']
    next_exp = j['data']['level_exp']['next_exp']
    sub_exp = int(next_exp) - int(current_exp)
    if coin_operated:
        days = int(int(sub_exp) / 65)
    else:
        days = int(int(sub_exp) / 15)
    coin = getCoin()
    msg = f"欢迎 {username}! 当前等级：" + str(level) + " ,当前经验：" + \
          str(current_exp) + ",升级还差 " + str(sub_exp) + \
          "经验，还需" + str(days) + "天，" + "当前硬币数量：" + str(coin)
    return current_exp, msg


# 推荐动态
def getActiveInfo():
    url = "http://api.bilibili.com/x/web-interface/archive/related?aid=" + \
          str(7)
    cookie = extract_cookies(cookies)
    r = requests.get(url, cookies=cookie).text
    j = json.loads(r)
    return j


# 推荐动态第二种方式
def getVideo():
    random_page = random.randint(0, 167)
    url = "http://api.bilibili.cn/recommend?page=" + str(random_page)
    cookie = extract_cookies(cookies)
    r = requests.get(url, cookies=cookie).text
    j = json.loads(r)
    return j


# 投币 分享5次
def Task():
    coin_num = getCoin()
    coin_count = 0
    for i in range(0, 10):
        j = getVideo()
        list_len = len(j['list'])
        random_list = random.randint(1, list_len - 1)
        bvid = j['list'][random_list]['bvid']
        aid = j['list'][random_list]['aid']
        title = j['list'][random_list]['title']
        print("第 {} 个视频:{} ==== ({}, {})".format(i+1,title,bvid,aid))
        toview(bvid)
        time.sleep(3)
        shareVideo(bvid)
        time.sleep(3)
        if coin_operated:
            while coin_count < min(coin_num, 5):
                coin_code = tocoin(bvid)
                if coin_code == 1:
                    coin_count += 1
                elif coin_code == -99:
                    return
        else:
            pass
        print('======================================')


# 观看视频【不会点赞投币】
def toview(bvid):
    playedTime = random.randint(10, 100)
    url = "https://api.bilibili.com/x/click-interface/web/heartbeat"
    data = {
        'bvid': bvid,
        'played_time': playedTime,
        'csrf': csrf
    }
    cookie = extract_cookies(cookies)
    j = requests.post(url, data=data, cookies=cookie).json()
    code = j['code']
    if code == 0:
        print('看视频成功!')
    else:
        print('看视频失败!')


# 分享视频
def shareVideo(bvid):
    url = "https://api.bilibili.com/x/web-interface/share/add"
    data = {
        'bvid': bvid,
        'csrf': csrf
    }
    cookie = extract_cookies(cookies)
    # 需要请求头
    header = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36 Edg/93.0.961.38",
    }
    r = requests.post(url, data=data, cookies=cookie, headers=header).text
    j = json.loads(r)
    code = j['code']
    if code == 0:
        print('分享成功!')
    else:
        print('分享失败!')


# 投币函数
def tocoin(bvid):
    coinNum = getCoin()
    if coinNum == 0:
        print('太穷了，硬币不够!')
        return -99
    url = "http://api.bilibili.com/x/web-interface/coin/add"
    data = {
        'bvid': bvid,
        'multiply': 1,
        'select_like': 1,
        'csrf': csrf
    }
    cookie = extract_cookies(cookies)
    r = requests.post(url, data=data, cookies=cookie).text
    j = json.loads(r)
    code = j['code']
    if code == 0:
        print(str(bvid) + ' 投币成功!')
        return 1
    else:
        print(str(bvid) + ' 投币失败!')
        return 0


# 开启任务运行
def run():
    # 当前经验
    exp1, msg = getInfo()
    print(msg)
    Task()
    exp2, c = getInfo()
    print(f'任务完成，获得经验{int(exp2) - int(exp1)}')


# 云函数使用
def main_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    print("Received context: " + str(context))
    run()
    return "------ end ------"


if __name__ == '__main__':
    run()
