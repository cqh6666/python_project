from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import open3d as o3d

import meshio  # 读取三维mesh的库

OBJ_file = 'matteonormb.obj'
PLY_file = 'matteonormb.ply'


def zero_mean(data_matrix):
    """
    中心化
    :param data_matrix: 数据矩阵 nxd
    :return: newData, meanVal
    """
    meanVal = np.mean(data_matrix, axis=0)  # 按列求均值，即求各个特征的均值
    data_new_matrix = data_matrix - meanVal
    return data_new_matrix, meanVal


def cov_mat(data_new_matrix):
    """
    求协方差矩阵
    :param data_new_matrix:
    :return:
    """
    covMat = np.cov(data_new_matrix, rowvar=0)
    return covMat


def lin_eig(con_matrix):
    """
    求特征值和特征矩阵
    :param con_matrix:协方差矩阵
    :return:
    """
    eigVals, eigVects = np.linalg.eig(np.mat(con_matrix))
    return eigVals, eigVects


def select_eig(data_matrix, eigVals, eigVects, d_l):
    data_new_matrix, meanVal = zero_mean(data_matrix)
    eigValIndice = np.argsort(-eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[0:d_l:1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    lowDDataMat = data_new_matrix * n_eigVect  # 低维特征空间的数据
    # reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
    return lowDDataMat


def prim_com_analy(points, n_components):
    """
    主成分分析PCA
    流程：
    1. 中心化
    2. 协方差矩阵X * X^T
    3. 做特征值分解
    4. 取最大的d'个特征值所对应的特征向量 w1,w2,w3,..wd.
    :param
    data_set : 数据集
    n_components:低维度
    :return:
    """
    # 去中心化
    data_new_matrix, meanVal = zero_mean(points)
    # 求协方差矩阵
    covMat = cov_mat(data_new_matrix)
    # 特征值分解，求特征值和特征向量
    eigVals, eigVects = lin_eig(con_matrix=covMat)
    # 选取最大的d_l个特征值对应的特征向量,lowDDataMat 降维的矩阵，reconMat 降维且经过变换的投影矩阵
    lowDDataMat = select_eig(points, eigVals, eigVects, n_components)
    return lowDDataMat


def load_xyz(file):
    xx = []
    yy = []
    zz = []
    for line in open(file):
        points = line.split(' ')
        point_z = points[3].replace('\n', '')
        xx.append(float(points[1]))
        yy.append(float(points[2]))
        zz.append(float(point_z))

    # print(xx,yy,zz)
    return xx, yy, zz


def plot_3D(file):
    """
    绘画3D点
    :param file: obj
    :return:
    """
    fig = plt.figure()
    ax1 = Axes3D(fig)
    p_x, p_y, p_z = load_xyz(file)
    ax1.scatter(p_x, p_y, p_z)  # 绘制散点图
    plt.show()


def dim_reduct_plot(file):
    """
    获取 matteonormb.obj 的所有点，降噪后对其进行绘图
    :param file: obj
    :return:
    """
    mesh = meshio.read(file)
    points = mesh.points

    lowDDataMat = prim_com_analy(points, 2)
    lowData = np.array(lowDDataMat)

    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(points)

    mds = MDS(n_components=2)
    X_mds = mds.fit_transform(points)

    plt.subplot(121)
    plt.scatter(lowData[:, 0], lowData[:, 1], marker='o')
    plt.subplot(122)
    plt.scatter(X_mds[:, 0], X_mds[:, 1], marker='o')
    plt.show()


def draw_o3d(file):
    """
    visualization of point clouds.
    :param file: ply
    :return:
    """
    pcd = o3d.io.read_point_cloud(file)
    o3d.visualization.draw_geometries([pcd], width=1000, height=1008)


dim_reduct_plot(OBJ_file)
