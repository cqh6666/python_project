#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   FrequencyDomainFiltering.py
@Time    :   2021/06/04 18:09:50
@Author  :   Alex Chan 
@Version :   1.0
@Contact :   2018ch@m.scnu.edu.cn
@Desc    :   None
'''
''' 频域滤波实验要求
1) processing any size image from an image file f(x,y)
2) dealing with any size transfer filter H(u,v) (lowpass or highpass filter)
3) delivering the inverse Fourier Transformation after filtering
4) showing your processed image with the same size to its original one.
5) showing some examples from your processing codes and giving your comments about
them.

所以对于程序来说
1. 对于输入图像大小进行扩大两倍的大小，图像不动
2. 进行傅里叶变换
3. 低频的位置变换移动到中心
2. 生成一个低通或高通滤波器 挪到左上角
3. 进行逆向傅里叶变换
'''

# here put the import lib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class FreDoFilter:
    def __init__(self, input_image, filter_type=0,filter_size=80) -> None:
        self.input_image = input_image
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.image_shpae = 0
        self.process_image = []

    def process_size(self):
        img = Image.open(self.input_image).convert('L')
        img_array = np.array(img)
        self.process_image.append(img_array)
        self.image_shpae = img_array.shape
        image_rows, image_cols = img_array.shape
        img_new_array = np.pad(img_array, ((0, image_rows), (0, image_cols)),
                               'constant',
                               constant_values=0)
        self.process_image.append(img_new_array)
        img_new_array2 = img_new_array.copy()
        image_new_rows, image_new_cols = img_new_array.shape
        for i in range(image_new_rows):
            for j in range(image_new_cols):
                temp = np.power(-1, i + j)
                if (temp < 0):
                    img_new_array2[i, j] = 0
                else:
                    img_new_array2[i, j] *= temp
        
        self.process_image.append(img_new_array2)
        self.input_image = img_new_array2

    def transfer_filter(self):
        # dft变换
        dft = np.fft.fft2(np.float32(self.input_image))

        # 添加滤波
        fshift = np.fft.fftshift(dft)
        self.process_image.append(np.log(np.abs(fshift)))
        fshift *= self.wave_filter()
        # 移回左上角 逆傅里叶变换
        ishift = np.fft.ifftshift(fshift)
        self.process_image.append(np.log(np.abs(ishift)))

        new_img = np.abs(np.fft.ifft2(ishift))

        self.process_image.append(new_img)

        # 重新截取图像大小
        new_img = new_img[:self.image_shpae[0],:self.image_shpae[1]]
        self.process_image.append(new_img)
        self.input_image = new_img

    def wave_filter(self):
       
        filter_shape = (self.image_shpae[0]*2, self.image_shpae[1]*2)
        
        if self.filter_type == 0:
            # 低通滤波器
            filters = np.zeros(filter_shape,dtype=np.uint8)
            filters[self.image_shpae[0]-self.filter_size:self.image_shpae[0]+self.filter_size,self.image_shpae[1]-self.filter_size:self.image_shpae[1]+self.filter_size] = 1
            return filters
        elif self.filter_type == 1:
            # 高通滤波器
            filters = np.ones(filter_shape,dtype=np.uint8)
            filters[self.image_shpae[0]-self.filter_size:self.image_shpae[0]+self.filter_size,self.image_shpae[1]-self.filter_size:self.image_shpae[1]+self.filter_size] = 0
            return filters
        elif self.filter_type == 2:
            # 带通滤波器
            filters1 = np.ones(filter_shape,dtype=np.uint8)
            filters1[self.image_shpae[0]-self.filter_size:self.image_shpae[0]+self.filter_size,self.image_shpae[1]-self.filter_size:self.image_shpae[1]+self.filter_size] = 0
            filters2 = np.zeros(filter_shape,dtype=np.uint8)
            filters2[self.image_shpae[0]-self.filter_size*10:self.image_shpae[0]+self.filter_size*10,self.image_shpae[1]-self.filter_size*10:self.image_shpae[1]+self.filter_size*10] = 1
            return filters1*filters2
        else:
            print("类型不存在...")
            return -1

    def show_image(self):
        plt.figure(figsize=(20,10))
        for i,img in enumerate(self.process_image):
            plt.subplot(241+i)
            plt.imshow(self.process_image[i],cmap='gray')
            plt.title(i)
        
        plt.show()
        

    def fit(self):
        self.process_size()
        self.transfer_filter()
        self.show_image()

    


if __name__ == '__main__':
    input_image = "./images/example.jpg"
    output = FreDoFilter(input_image,2,10).fit()
