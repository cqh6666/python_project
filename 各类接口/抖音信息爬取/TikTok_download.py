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

import requests, re, json, time
import os
from urllib import parse

base_headers = {
    'cookie': 'MONITOR_WEB_ID=87378296-c50a-487e-bff8-e3aecbf7af1a; _tea_utm_cache_1243={'
              '%22utm_source%22:%22copy%22%2C%22utm_medium%22:%22android%22%2C%22utm_campaign%22:%22client_share%22}',
    'referer': 'https://www.iesdouyin.com/share/user/50904259373?did'
               '=MS4wLjABAAAARClYU7g5kQpsKl_FGkbX_H4RqhOI3tEocOouVMdV64AbFn8JbB4e4owdO2fMi0MG&iid'
               '=MS4wLjABAAAAg1jAdyi3XsD26Z6Jcr0TwFLdlILvnCfyovnc_k8-DYs&with_sec_did=1&u_code=jb3bb1ji&sec_uid'
               '=MS4wLjABAAAA070w5X9l5I82jsuGY6ntBMGlOYp8yzp4-rH8X1qCEPw&timestamp=1619845930&utm_source=copy'
               '&utm_campaign=client_share&utm_medium=android&share_app_name=douyin',
    'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) '
                  'Version/13.0.3 Mobile/15E148 Safari/604.1 Edg/90.0.4430.85 '
}


def is_exists_dir(user_list):
    """
    根据用户名创建新的文件夹
    :param dy_url:
    :return:
    """
    author = user_list[0]['author_name']

    dir_path = "./dy_{}".format(author)
    is_exists = os.path.exists(dir_path)
    if not is_exists:
        os.makedirs(dir_path)
        print("创建 - {} - 文件夹成功...".format(dir_path))
    else:
        print("已存在 - {} - 文件夹 ...".format(dir_path))
    return dir_path


def save_info_json(user_list, video_list):
    """
    保存为json格式的文件
    :param info_list:
    :param file_name:
    :return:
    """
    all_list = [user_list, video_list]
    # 保存的文件夹
    save_path = is_exists_dir(user_list)
    author = user_list[0]['author_name']
    file_name = "dy_{}.json".format(author)
    # 格式化字典
    json_str = json.dumps(all_list, indent=4, ensure_ascii=False)

    with open(os.path.join(save_path, file_name), 'w+', encoding='utf-8') as json_file:
        json_file.write(json_str)
        print("保存 - {} - 成功".format(json_file))


def get_url_list(string):
    """
    返回解码后的url
    :param string:
    :return:
    """
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return url


class TikTok():
    def __init__(self, headers=base_headers):
        self.headers = headers

    def get_url_msg(self, dy_url, subject="user", mode='post', count=50):
        """
        url有两种，一种是用户url，一种是视频url
        :param count:
        :param subject:
        :param mode:
        :param dy_url:
        :return:返回得到的所有信息
        """
        # 获取解码的url
        res_url = requests.get(dy_url).url
        # 返回url的所有参数
        params = parse.parse_qs(parse.urlparse(res_url).query)
        print('params',params.keys())
        # 官方api
        api_post_url = ""
        if subject == "user":
            sec_uid = params['sec_uid'][0]
            api_post_url = "https://www.iesdouyin.com/web/api/v2/user/info/?sec_uid={}".format(sec_uid)
        elif subject == "all_video":
            sec_uid = params['sec_uid'][0]
            api_post_url = 'https://www.iesdouyin.com/web/api/v2/aweme/%s/?sec_uid=%s&count=%s&max_cursor=0&aid=1128&_signature=7GuRDgAAjOgEsbNzyyuYwuxrkR&dytk=' % (
                mode, sec_uid, str(count))
        elif subject == "video":
            video_id = re.findall('video/(\d+)/', res_url)[0]
            api_post_url = "https://www.iesdouyin.com/web/api/v2/aweme/iteminfo/?item_ids={}".format(video_id)

        res_post = requests.get(url=api_post_url, headers=self.headers)

        res_msg = []
        # 访问成功
        if res_post.status_code == 200:
            print("成功获得该用户的所有信息！")
            res_html = json.loads(res_post.content.decode('utf-8'))
            print(res_html)
            # 返回的所有信息
            if subject == "user":
                res_msg.append(res_html['user_info'])
            elif subject == "all_video":
                res_msg = res_html['aweme_list']
            elif subject == "video":
                print(res_html)
                res_msg = res_html['item_list']
        else:
            print('查不到此链接的信息！')

        return res_msg

    def get_user_info(self, dy_url):
        """
        爬取用户的所有信息
        :param dy_url:
        :return:
        """
        res_msg = self.get_url_msg(dy_url, subject="user")[0]

        user_info = {}
        if res_msg is not None:
            user_info['uid'] = res_msg["uid"]
            user_info['dy_id'] = res_msg['unique_id']
            user_info['author_name'] = res_msg['nickname']
            user_info['signature'] = res_msg['signature']

            user_info['avatar_url'] = res_msg['avatar_larger']['url_list'][0]
            user_info['aweme_count'] = res_msg['aweme_count']
            user_info['favoriting_count'] = res_msg['favoriting_count']

            user_info['follower_count'] = res_msg['follower_count']
            user_info['following_count'] = res_msg['following_count']
            user_info['total_favorited'] = res_msg['total_favorited']

        user_list = [user_info]

        return user_list

    def get_all_video_info(self, dy_url):
        """
        得到用户所有视频信息
        :return:
        """
        res_msg = self.get_url_msg(dy_url, subject="all_video")

        # 所有视频信息
        video_list = []
        len_res_msg = len(res_msg)
        for i in range(len_res_msg):
            video_info = {}
            try:
                video_info["author_id"] = res_msg[i]['author']['short_id']
                video_info["author_name"] = res_msg[i]['author']['nickname']
                video_info["video_id"] = res_msg[i]['aweme_id']
                video_info["desc"] = res_msg[i]['desc']

                statistic = {
                    "comment_count": res_msg[i]['statistics']['comment_count'],
                    "digg_count": res_msg[i]['statistics']['digg_count'],
                    "share_count": res_msg[i]['statistics']['share_count']
                }

                video_info["statistic"] = statistic
                video_info['tag'] = [tag['hashtag_name'] for tag in res_msg[i]['text_extra']]

                video_info["dynamic_cover"] = res_msg[i]['video']['dynamic_cover']['url_list'][0]
                video_info["video_url"] = res_msg[i]['video']['play_addr']['url_list']
                video_list.append(video_info)
            except:
                print('第{}次爬取出现了问题'.format(i + 1))

        return video_list

    def get_one_video_info(self, dy_url):
        """
        获得单个视频的详细信息
        :param dy_url:
        :return:
        """
        res_msg = self.get_url_msg(dy_url, subject="video")[0]
        video_info = {}
        try:
            video_info['author_user_id'] = res_msg['author_user_id']
            video_info['aweme_id'] = res_msg['aweme_id']
            video_info['desc'] = res_msg['desc']
            video_info['create_time'] = res_msg['create_time']
            video_info['author_name'] = res_msg['author']['nickname']
            video_statistics = {
                'comment_count': res_msg['statistics']['comment_count'],
                'digg_count': res_msg['statistics']['digg_count'],
                'share_count': res_msg['statistics']['share_count'],
            }

            video_music = {
                'music_id': res_msg['music']['id'],
                'music_url': res_msg['music']['play_url']['url_list'][0],
                'music_title': res_msg['music']['title']
            }
            video_info['music'] = video_music
            video_info['statistics'] = video_statistics
            video_info['dynamic_cover_url'] = res_msg['video']['dynamic_cover']['url_list'][0]
            video_info['video_url'] = str(res_msg["video"]["play_addr"]["url_list"][0]).replace('playwm', 'play')
        except:
            print("获取视频信息失败！！！")

        return video_info

    def download_all_video(self, dy_url, video_down=True, dynamic_down=False):
        """
        获取用户主页所有视频链接
        :param video_down:
        :param dynamic_down:
        :param video_list:
        :return:
        """
        video_list = self.get_all_video_info(dy_url)
        author = video_list[0]['author_name']
        dir_path = is_exists_dir(video_list)

        try:
            for video in video_list:

                # 若要下载视频
                if video_down:
                    for v_url in video['video_url']:
                        video_res = requests.get(v_url, headers=self.headers)
                        # 保存视频
                        file_name = "{}-{}.mp4".format(author, video['desc'])
                        with open(os.path.join(dir_path, file_name), 'wb') as f:
                            f.write(video_res.content)
                            print("{}的视频:{} - 下载完成...".format(author, video['desc']))
                            break

                if dynamic_down:
                    dynamic_res = requests.get(video['cover_picture'], headers=self.headers)
                    # 保存封面
                    file_name = "{}-{}.jpg".format(author, video['desc'])
                    with open(os.path.join(dir_path, file_name), 'wb') as f:
                        f.write(dynamic_res.content)
                        print("{}的封面:{} - 下载完成...".format(author, video['desc']))
        except:
            print("下载 {} 的 视频/封面 失败!!!".format(author))

        print('{}的视频下载完成，共计{}个!'.format(author, len(video_list)))

    def download_one_video(self, dy_url, video_down=True, music_down=False, dynamic_down=False):
        """
        下载单个视频
        :param music_down:
        :param dy_url:
        :param video_down:
        :param dynamic_down:
        :return:
        """

        video_info = self.get_one_video_info(dy_url)
        author_name = video_info['author_name']
        desc = video_info['desc']
        try:
            # 若要下载视频
            if video_down:
                video_url = video_info['video_url']
                print('无水印视频链接为:', video_url)
                video_res = requests.get(video_url, headers=self.headers)
                # 保存视频
                file_name = "{}-{}.mp4".format(author_name, desc)
                with open(file_name, 'wb') as f:
                    f.write(video_res.content)
                    print("{}的视频:{} - 下载完成...".format(author_name, desc))

            if music_down:
                music_url = video_info['music']['music_url']
                print('音乐链接为:', music_url)
                music_res = requests.get(music_url, headers=self.headers)
                # 保存封面
                file_name = "{}-{}.mp3".format(author_name, desc)
                with open(file_name, 'wb') as f:
                    f.write(music_res.content)
                    print("{}的音乐:{} - 下载完成...".format(author_name, desc))

            if dynamic_down:
                dynamic_url = video_info['dynamic_cover_url']
                dynamic_res = requests.get(dynamic_url, headers=self.headers)
                # 保存封面
                file_name = "{}-{}.jpg".format(author_name, desc)
                with open(file_name, 'wb') as f:
                    f.write(dynamic_res.content)
                    print("{}的封面:{} - 下载完成...".format(author_name, desc))
        except:
            print("下载 {} 的视频失败!!!".format(author_name))


one_video_url = 'https://v.douyin.com/eShjkxS/'  # 单个视频


def save_json(tiktok, video_url):
    user_list = tiktok.get_user_info(video_url)
    video_list = tiktok.get_all_video_info(video_url)
    save_info_json(user_list, video_list)


def test():
    api_post_url = 'https://jokeai.zongcaihao.com/douyin/v292/comment/list?aweme_id=6947627831081864482&cursor=0'
    res_post = requests.get(url=api_post_url, headers=base_headers)
    res_html = json.loads(res_post.content.decode('utf-8'))
    return res_html


if __name__ == "__main__":
    strings = "https://v.douyin.com/eSkhPvm/ ,https://v.douyin.com/eSkhL1o/, https://v.douyin.com/eSkaDe4/"
    all_video_url = get_url_list(strings)[0]
    postParams = {
        "user_id": '59189910855',
        'count': '21',
        'max_cursor': '1',
        'min_cursor': '0',
        'aid': '1128'

    }
    res = requests.get("https://aweme.snssdk.com/aweme/v1/aweme/favorite/?", params=postParams,  headers=base_headers)

    print(res.content)
    # tk = TikTok()
    # msg = tk.get_url_msg(dy_url=all_video_url[0], subject="all_video", mode="like")
    # print(msg)
    # for video_url in all_video_url:
    #     tk = TikTok()
    #     save_json(tk, video_url)
    #     tk.download_all_video(video_url)
    # print(test())
