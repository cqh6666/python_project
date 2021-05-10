# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     BilateralFiltering
   Description :
   Author :       cqh
   date：          2021/5/7 16:17
-------------------------------------------------
   Change Activity:
                   2021/5/7:
-------------------------------------------------
"""
__author__ = 'cqh'

import cv2
import matplotlib.pyplot as plt
import numpy as np



# # apply guassian blur on src image
# img_gb = cv2.GaussianBlur(img, (51, 51), cv2.BORDER_DEFAULT)
# img_gb2 = cv2.blur(img, (51, 51), cv2.BORDER_DEFAULT)
# img_gb3 = cv2.GaussianBlur(img, (101, 101), cv2.BORDER_DEFAULT)
#
# # 展示不同的图片
# titles = ['img_gb', 'img_gb2', 'img_gb3']
# imgs = [img_gb, img_gb2, img_gb3]
#
# for i in range(len(imgs)):
#     # display input and output image
#     cv2.imshow("Gaussian Smoothing", np.hstack((img, imgs[i])))
#     cv2.waitKey(0)  # waits until a key is pressed
#     cv2.destroyAllWindows()  # destroys the window showing image
#
#


class BilateralFilter:
    def __init__(self, input_image, output_image, distance=5, sigma_color=0, sigma_space=0,
                 border_type=cv2.BORDER_DEFAULT):
        """
        构造函数
        :param input_image:
        :param output_image:
        :param distance:
        :param sigma_color:
        :param sigma_space:
        :param border_type:
        """
        self.distance = distance
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type
        self.input = input_image
        self.output = output_image

    def fit(self):
        """

        :return:
        """
        input_image = self.input
        assert (type(input_image) is np.ndarray), "读取不到图片信息..."     # 输入必须为图片格式

        image_shape = input_image.shape
        assert (image_shape[2] == 1 or image_shape[2] == 3), "图片通道错误..."    # 图片通道必须为1或3





    def test(self):
        return self.input.shape


if __name__ == '__main__':
    img = cv2.imread("1.jpg")
    bf = BilateralFilter(img, "output.jpg")
    print(bf.fit())
