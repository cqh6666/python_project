Help on module BilateralFiltering:

NAME
    BilateralFiltering

DESCRIPTION
    -------------------------------------------------
       File Name：     BilateralFiltering
       Description :
       Author :       cqh
       date：          2021/5/7 16:17
    -------------------------------------------------
       Change Activity:
                       2021/5/7:
    -------------------------------------------------

CLASSES
    builtins.object
        BilateralFilter
    
    class BilateralFilter(builtins.object)
     |  BilateralFilter(input_image, output_image, diameter=5, sigma_color=80.0, sigma_space=80.0)
     |  
     |  双边滤波类，通过fit()调用
     |  
     |  Methods defined here:
     |  
     |  __init__(self, input_image, output_image, diameter=5, sigma_color=80.0, sigma_space=80.0)
     |      构造函数
     |      :param input_image:输入图像的链接
     |      :param output_image:输出图像的名称
     |      :param diameter:像素邻域的直接
     |      :param sigma_color:颜色域的sigma值
     |      :param sigma_space:空间域的sigma值
     |  
     |  bilateral_filter(self, row, col, channel)
     |      对某个像素点的更新
     |      :param row:
     |      :param col:
     |      :param channel:
     |      :return:
     |  
     |  fit(self)
     |      遍历每个像素点
     |      :return:
     |  
     |  test(self)
     |      调用opencv下自带的双边滤波
     |      :return:
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    compare_diameter(img_url, diameters)
        比较不同直径下的双边滤波的影响
        :param img_url:
        :param diameters:
        :return:
    
    compare_noise(img_url)
        比较不同噪声下的双边滤波的作用
        :param img_url:
        :return:
    
    compare_sigma(img_url, sigmas)
        比较不同sigma的影响
        :param img_url:
        :param sigmas: color or space
        :return:
    
    distance(i, j, m, n)
    
    gauss_noise(image, mean=0, var=0.001)
        添加高斯噪声
        image:原始图像
        mean : 均值
        var : 方差,越大，噪声越大
    
    gaussian_function(x, sigma)
        高斯函数，前面的系数可以不用计算
        :param x:
        :param sigma:
        :return:
    
    random_noise(image, noise_num)
        添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）
        :param image: 需要加噪的图片
        :param noise_num: 添加的噪音点数目，一般是上千级别的
        :return: img_noise
    
    sp_noise(image, prob)
        添加椒盐噪声
        image:原始图片
        prob:噪声比例

AUTHOR
    cqh

FILE
    d:\python_project\图像处理\滤波\bilateralfiltering.py

