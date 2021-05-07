# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     pic_crop
   Description :   截取饥荒的壁纸为 16:9
   Author :       cqh
   date：          2021/4/27 14:13
-------------------------------------------------
   Change Activity:
                   2021/4/27:
-------------------------------------------------
"""
__author__ = 'cqh'

from PIL import Image
import os
img_path = "pictures"


def image_crop_for_wallpaper():
    for file in os.listdir(img_path):
        img = Image.open(os.path.join(img_path,file))
        cropped = img.crop((0, 0, 1820, 1024))
        cropped.save("./updated_picture/{}".format(file))
