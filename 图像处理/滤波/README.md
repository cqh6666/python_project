Help on module BilateralFiltering:

NAME
    BilateralFiltering

DESCRIPTION
    -------------------------------------------------
       File Name��     BilateralFiltering
       Description :
       Author :       cqh
       date��          2021/5/7 16:17
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
     |  ˫���˲��࣬ͨ��fit()����
     |  
     |  Methods defined here:
     |  
     |  __init__(self, input_image, output_image, diameter=5, sigma_color=80.0, sigma_space=80.0)
     |      ���캯��
     |      :param input_image:����ͼ�������
     |      :param output_image:���ͼ�������
     |      :param diameter:���������ֱ��
     |      :param sigma_color:��ɫ���sigmaֵ
     |      :param sigma_space:�ռ����sigmaֵ
     |  
     |  bilateral_filter(self, row, col, channel)
     |      ��ĳ�����ص�ĸ���
     |      :param row:
     |      :param col:
     |      :param channel:
     |      :return:
     |  
     |  fit(self)
     |      ����ÿ�����ص�
     |      :return:
     |  
     |  test(self)
     |      ����opencv���Դ���˫���˲�
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
        �Ƚϲ�ֱͬ���µ�˫���˲���Ӱ��
        :param img_url:
        :param diameters:
        :return:
    
    compare_noise(img_url)
        �Ƚϲ�ͬ�����µ�˫���˲�������
        :param img_url:
        :return:
    
    compare_sigma(img_url, sigmas)
        �Ƚϲ�ͬsigma��Ӱ��
        :param img_url:
        :param sigmas: color or space
        :return:
    
    distance(i, j, m, n)
    
    gauss_noise(image, mean=0, var=0.001)
        ��Ӹ�˹����
        image:ԭʼͼ��
        mean : ��ֵ
        var : ����,Խ������Խ��
    
    gaussian_function(x, sigma)
        ��˹������ǰ���ϵ�����Բ��ü���
        :param x:
        :param sigma:
        :return:
    
    random_noise(image, noise_num)
        ��������㣨ʵ���Ͼ��������ͼ���Ͻ����ص�ĻҶ�ֵ��Ϊ255����ɫ��
        :param image: ��Ҫ�����ͼƬ
        :param noise_num: ��ӵ���������Ŀ��һ������ǧ�����
        :return: img_noise
    
    sp_noise(image, prob)
        ��ӽ�������
        image:ԭʼͼƬ
        prob:��������

AUTHOR
    cqh

FILE
    d:\python_project\ͼ����\�˲�\bilateralfiltering.py

