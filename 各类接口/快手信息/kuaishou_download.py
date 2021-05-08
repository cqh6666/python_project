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

import requests, json
from urllib import parse

base_headers = {
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, '
                  'like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1'
}


def save_info_json(ks, video_url):
    video_info = ks.get_one_video_info(video_url)
    file_name = "dy_{}.json".format(video_info['author_name'])
    json_str = json.dumps(video_info, indent=4, ensure_ascii=False)
    with open(file_name, 'w+', encoding='utf-8') as json_file:
        json_file.write(json_str)
        print("保存 - {} - 成功".format(json_file))


class KuaiShou():
    def __init__(self, headers=None):
        if headers is None:
            headers = base_headers
        self.headers = headers

    def get_url_msg(self, ks_url):
        """
        :param photo_id:
        :param ks_url:
        :return:
        """
        res_msg = {}
        try:
            url_res = requests.get(ks_url, headers=self.headers, allow_redirects=False)
            share_url = url_res.headers.get('Location')  # 取得原url
            params = parse.parse_qs(parse.urlparse(share_url).query)
            photo_id = params['photoId'][0]
            api_post = "https://video.kuaishou.com/graphql"
            new_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/90.0.4430.93 Safari/537.36 Edg/90.0.818.51',
                'content-type': "application/json",
                'Cookie': 'kpf=PC_WEB; kpn=KUAISHOU_VISION; clientid=3; didv=1620052138000; '
                          'did=web_16d4e83a73c842d08e66d1286833c2f5; '
                          'Hm_lvt_86a27b7db2c5c0ae37fee4a8a35033ee=1620052351; client_key=65890b29',
                'Referer': share_url
            }
            pay_load = {
                "operationName": "visionVideoDetail",
                "variables": {
                    "photoId": photo_id,
                    "page": "detail",
                },
                "query": "query visionVideoDetail($photoId: String, $type: String, $page: String, $webPageArea: "
                         "String) { "
                         "\n  visionVideoDetail(photoId: $photoId, type: $type, page: $page, webPageArea: "
                         "$webPageArea) { "
                         "\n    status\n    type\n    author {\n      id\n      name\n      following\n      "
                         "headerUrl\n "
                         "__typename\n    }\n    photo {\n      id\n      duration\n      caption\n      likeCount\n "
                         "realLikeCount\n      coverUrl\n      photoUrl\n      liked\n      timestamp\n      expTag\n "
                         "llsid\n      viewCount\n      videoRatio\n      stereoType\n      croppedPhotoUrl\n "
                         "manifest {\n        mediaType\n        businessType\n        version\n        adaptationSet "
                         "{\n "
                         "         id\n          duration\n          representation {\n            id\n            "
                         "defaultSelect\n            backupUrl\n            codecs\n            url\n            "
                         "height\n "
                         "width\n            avgBitrate\n            maxBitrate\n            m3u8Slice\n "
                         "     qualityType\n            qualityLabel\n            frameRate\n            featureP2sp\n    "
                         "        hidden\n            disableAdaptive\n            __typename\n          }\n          "
                         "__typename\n        }\n        __typename\n      }\n      __typename\n    }\n    tags {\n "
                         "type\n      name\n      __typename\n    }\n    commentLimit {\n      canAddComment\n      "
                         "__typename\n    }\n    llsid\n    __typename\n  }\n}\n "
            }
            pay_load_json = json.dumps(pay_load)

            api_res = requests.post(api_post, data=pay_load_json, headers=new_headers)
            res_html = json.loads(api_res.content.decode('utf-8'))
            res_msg = res_html['data'].get('visionVideoDetail')

        except requests.RequestException as e:
            print("get_url_msg() 获得信息出现了问题...", e)

        return res_msg

    def get_one_video_info(self, ks_url):
        """
        获得快手视频所有信息
        :param ks_url:
        :return:
        """
        res_msg = self.get_url_msg(ks_url)
        video_info = {}

        try:
            video_info['status_code'] = res_msg.get('status')
            video_info['author_id'] = res_msg['author'].get('id')
            video_info['author_name'] = res_msg['author'].get('name')
            video_info['aweme_id'] = res_msg['photo'].get('id')

            video_info['create_time'] = res_msg['photo'].get('timestamp')
            video_info['digg_count'] = res_msg['photo'].get('realLikeCount')
            video_info['view_count'] = res_msg['photo'].get('viewCount')

            video_info['cover_url'] = res_msg['photo'].get('coverUrl')
            video_info['video_url'] = res_msg['photo'].get('photoUrl')

            video_info['desc'] = res_msg['photo'].get('caption')
            if res_msg['tags'] is not None:
                video_info['tags'] = [tag.get('name') for tag in res_msg['tags']]
            else:
                video_info['tags'] = res_msg['tags']

        except:
            print("获取视频信息失败...")

        return video_info

    def download_one_video(self, ks_url, video_down=True, cover_down=False):
        video_info = ks.get_one_video_info(ks_url)

        if video_info['status_code'] != 1:
            print("获取视频信息失败...")
            return False

        author_name = video_info.get('author_name')
        desc = video_info.get('desc')
        # 保存视频

        try:
            if video_down:
                video_url = video_info['video_url']
                # print('无水印视频链接为:', video_url)
                video_res = requests.get(video_url, headers=self.headers)
                file_name = "{}-{}.mp4".format(author_name, desc)
                with open(file_name, 'wb') as f:
                    f.write(video_res.content)
                    print("{}的视频:{} - 下载完成...".format(author_name, desc))

            if cover_down:
                cover_url = video_info['cover_url']
                cover_res = requests.get(cover_url, headers=self.headers)
                file_name = "{}-{}.jpg".format(author_name, desc)
                with open(file_name, 'wb') as f:
                    f.write(cover_res.content)
                    print("{}的封面:{} - 下载完成...".format(author_name, desc))
        except requests.RequestException as e:
            print("download_one_video()出现了问题...", e)

        return True

    def get_comment_info(self, ks_url):
        """
        ajax动态请求，pcursor的规律不太懂，暂时没办法，只能取到第一页。
        :param ks_url:
        :return:
        """
        url_res = requests.get(ks_url, headers=self.headers, allow_redirects=False)
        share_url = url_res.headers.get('Location')  # 取得原url
        params = parse.parse_qs(parse.urlparse(share_url).query)
        photo_id = params['photoId'][0]

        api_post = "https://video.kuaishou.com/graphql"

        new_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/90.0.4430.93 Safari/537.36 Edg/90.0.818.51',
            'content-type': "application/json",
            'Cookie': 'kpf=PC_WEB; kpn=KUAISHOU_VISION; clientid=3; didv=1620052138000; '
                      'did=web_16d4e83a73c842d08e66d1286833c2f5; Hm_lvt_86a27b7db2c5c0ae37fee4a8a35033ee=1620052351; '
                      'client_key=65890b29',
            'Referer': share_url
        }
        pay_load = {
            "operationName": "commentListQuery",
            "variables": {
                "photoId": photo_id,
                "pcursor": "",
            },
            "query": "query commentListQuery($photoId: String, $pcursor: String) {\n  visionCommentList(photoId: "
                     "$photoId, pcursor: $pcursor) {\n    commentCount\n    pcursor\n    rootComments {\n      "
                     "commentId\n      authorId\n      authorName\n      content\n      headurl\n      timestamp\n    "
                     "  likedCount\n      realLikedCount\n      liked\n      status\n      subCommentCount\n      "
                     "subCommentsPcursor\n      subComments {\n        commentId\n        authorId\n        "
                     "authorName\n        content\n        headurl\n        timestamp\n        likedCount\n        "
                     "realLikedCount\n        liked\n        status\n        replyToUserName\n        replyTo\n       "
                     " __typename\n      }\n      __typename\n    }\n    __typename\n  }\n}\n "
        }
        pay_load_json = json.dumps(pay_load)
        api_res = requests.post(api_post, data=pay_load_json, headers=new_headers)
        res_html = json.loads(api_res.content.decode('utf-8'))
        res_msg = res_html['data']['visionCommentList']
        return res_msg


if __name__ == "__main__":
    user_url = "https://v.kuaishou.com/aOgset"
    ks = KuaiShou()
    info = ks.get_url_msg(user_url)
    print(info)
