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

import random
import cv2
import numpy as np
import datetime


class BilateralFilter:
    """
    双边滤波类，通过fit()调用
    """

    def __init__(self, input_image, output_image, diameter=5, sigma_color=80.0, sigma_space=80.0):
        """
        构造函数
        :param input_image:输入图像的链接
        :param output_image:输出图像的名称
        :param diameter:像素邻域的直接
        :param sigma_color:颜色域的sigma值
        :param sigma_space:空间域的sigma值
        """
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.input_image = input_image
        self.output_image = output_image

    def bilateral_filter(self, row, col, channel):
        """
        对某个像素点的更新
        :param row:
        :param col:
        :param channel:
        :return:
        """
        # 找到中心点
        center = self.diameter // 2
        shape = self.input_image.shape
        # 权重，像素点初始化
        W_p = 0
        filter_pixel = 0

        for i in range(self.diameter):
            for j in range(self.diameter):
                # 直径内的遍历
                neighbor_i = row - center + i
                neighbor_j = col - center + j

                # 处理边缘 >len
                if neighbor_i >= shape[0]:
                    neighbor_i -= shape[0]
                if neighbor_j >= shape[1]:
                    neighbor_j -= shape[1]
                # <0 不用处理

                # 双边的高斯
                # 空间域
                g_s = gaussian_function(distance(neighbor_i, neighbor_j, row, col), self.sigma_space)
                # 颜色域，需要转化为int，因为原类型只能在[0,255]范围内
                g_r = gaussian_function(
                    int(self.input_image[neighbor_i][neighbor_j][channel]) - int(self.input_image[row][col][channel]),
                    self.sigma_color
                )

                w = g_s * g_r
                filter_pixel += w * self.input_image[neighbor_i][neighbor_j][channel]  # 过滤后的像素点
                W_p += w

        filter_pixel = int(filter_pixel / W_p)
        return filter_pixel

    def fit(self):
        """
        遍历每个像素点
        :return:
        """
        input_image = self.input_image
        assert (type(input_image) is np.ndarray), "读取不到图片信息..."  # 输入必须为图片格式

        image_shape = input_image.shape
        assert (image_shape[2] == 1 or image_shape[2] == 3), "图片通道错误..."  # 图片通道必须为1或3

        filter_image = np.zeros(image_shape)
        for channel_i in range(image_shape[2]):
            for row_i in range(image_shape[0]):
                for col_i in range(image_shape[1]):
                    filter_image[row_i][col_i][channel_i] = self.bilateral_filter(row_i, col_i, channel_i)
                print(f"成功计算第{channel_i}层的{row_i}行像素...")

        filter_image = filter_image.astype(np.uint8)
        # 保存图片
        cv2.imwrite(self.output_image, filter_image)
        return filter_image

    def test(self):
        """
        调用opencv下自带的双边滤波
        :return:
        """
        blur = cv2.bilateralFilter(self.input_image, 30, 50, 50)
        cv2.imshow("bilateral:", blur)
        cv2.waitKey(0)


def gauss_noise(image, mean=0, var=0.001):
    """
        添加高斯噪声
        image:原始图像
        mean : 均值
        var : 方差,越大，噪声越大
    """
    image = cv2.imread(image)
    image = cv2.resize(image, (128, 128))

    image = np.array(image / 255, dtype=float)  # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    noise = np.random.normal(mean, var ** 0.5, image.shape)  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    out = image + noise  # 将噪声和原始图像进行相加得到加噪后的图像
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)  # clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
    out = np.uint8(out * 255)  # 解除归一化，乘以255将加噪后的图像的像素值恢复
    # cv.imshow("gasuss", out)
    noise = noise * 255
    return [noise, out]


def sp_noise(image, prob):
    """
    添加椒盐噪声
    image:原始图片
    prob:噪声比例
    """
    image = cv2.imread(image)
    image = cv2.resize(image, (128, 128))

    output = np.zeros(image.shape, np.uint8)
    noise_out = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()  # 随机生成0-1之间的数字
            if rdn < prob:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                output[i][j] = 0
                noise_out[i][j] = 0
            elif rdn > thres:  # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                output[i][j] = 255
                noise_out[i][j] = 255
            else:
                output[i][j] = image[i][j]  # 其他情况像素点不变
                noise_out[i][j] = 100
    result = [noise_out, output]  # 返回椒盐噪声和加噪图像
    return result


def random_noise(image, noise_num):
    """
    添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）
    :param image: 需要加噪的图片
    :param noise_num: 添加的噪音点数目，一般是上千级别的
    :return: img_noise
    """
    #
    # 参数image：，noise_num：
    img = cv2.imread(image)
    img = cv2.resize(img, (128, 128))

    img_noise = img
    # cv2.imshow("src", img)
    rows, cols, chn = img_noise.shape
    # 加噪声
    for i in range(noise_num):
        x = np.random.randint(0, rows)  # 随机生成指定范围的整数
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise


def gaussian_function(x, sigma):
    """
    高斯函数，前面的系数可以不用计算
    :param x:
    :param sigma:
    :return:
    """
    return np.exp(- (np.power(x, 2) / (2 * np.power(sigma, 2))))


def distance(i, j, m, n):
    return np.sqrt(np.power(i - m, 2) + np.power(j - n, 2))


def compare_diameter(img_url, diameters):
    """
    比较不同直径下的双边滤波的影响
    :param img_url:
    :param diameters:
    :return:
    """
    images = []
    input_img = cv2.imread(img_url)  # (399, 400, 3)
    input_img = cv2.resize(input_img, (128, 128))
    begin_time = datetime.datetime.now()
    for diameter in diameters:
        bf = BilateralFilter(input_img, f"output_diameter_{diameter}.jpg", diameter=diameter)
        output_image = bf.fit()
        images.append(output_image)

    last_time = datetime.datetime.now()
    plus_time = (last_time - begin_time).seconds
    print(f"共计运行了{plus_time}秒.")
    com_dia_bf_img = np.hstack((input_img, images[0], images[1], images[2]))
    cv2.imwrite("com_dia_bf_img.jpg", com_dia_bf_img)
    cv2.imshow("Difference : original - diameters[3 10 20]", com_dia_bf_img)
    cv2.waitKey(0)


def compare_noise(img_url):
    """
    比较不同噪声下的双边滤波的作用
    :param img_url:
    :return:
    """
    begin_time = datetime.datetime.now()

    input_img = cv2.imread(img_url)  # (399, 400, 3)
    input_img = cv2.resize(input_img, (128, 128))

    random_noise_img = random_noise(img_url, 500)
    _, sp_noise_img = sp_noise(img_url, 0.01)
    _, ga_noise_img = gauss_noise(img_url, 0, 0.005)

    com_noise_img = np.hstack((input_img, random_noise_img, sp_noise_img, ga_noise_img))
    cv2.imwrite("com_noise_img.jpg", com_noise_img)
    cv2.imshow("Difference : original - noise", com_noise_img)
    cv2.waitKey(0)

    noise_imgs = [random_noise_img, sp_noise_img, ga_noise_img]
    noises = ["random_noise", "sp_noise", "gauss_noise"]
    images = []
    for name, img in zip(noises, noise_imgs):
        bf = BilateralFilter(img, f"output_noise_{name}.jpg", diameter=15)
        output_image = bf.fit()
        images.append(output_image)
    last_time = datetime.datetime.now()
    plus_time = (last_time - begin_time).seconds
    print(f"共计运行了{plus_time}秒.")
    com_noise_bf_img = np.hstack((input_img, images[0], images[1], images[2]))
    cv2.imwrite("com_noise_bf_img.jpg", com_noise_bf_img)
    cv2.imshow("Difference : original - bf_noises", com_noise_bf_img)
    cv2.waitKey(0)


def compare_sigma(img_url, sigmas):
    """
    比较不同sigma的影响
    :param img_url:
    :param sigmas: color or space
    :return:
    """
    images = []
    input_img = cv2.imread(img_url)  # (399, 400, 3)
    input_img = cv2.resize(input_img, (128, 128))
    begin_time = datetime.datetime.now()
    for sigma in sigmas:
        bf = BilateralFilter(input_img, f"output_sigma_{sigma}.jpg", sigma_space=sigma)
        output_image = bf.fit()
        images.append(output_image)

    last_time = datetime.datetime.now()
    plus_time = (last_time - begin_time).seconds
    print(f"共计运行了{plus_time}秒.")
    com_sigma_space_bf_img = np.hstack((input_img, images[0], images[1], images[2]))
    cv2.imwrite("com_sigma_space_bf_img.jpg", com_sigma_space_bf_img)
    cv2.imshow("Difference : original - sigma_space[10 50 150]", com_sigma_space_bf_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_url = "2.jpg"
    input_img = cv2.imread(img_url)
    input_img = cv2.resize(input_img, (128, 128))

    # 正常运行
    bf = BilateralFilter(input_img, "main_output.jpg", diameter=5, sigma_space=50, sigma_color=50)
    bf.fit()

    # 比较不同参数
    # diameters = [3, 10, 25]
    # sigmas = [10, 50, 150]
    # compare_diameter(img_url, diameters)
    # compare_noise(img_url)
    # compare_sigma(img_url, sigmas)
