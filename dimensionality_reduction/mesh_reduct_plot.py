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


def get_distance_matrix(data):
    """
    求距离矩阵
    :param data: n x m 矩阵数据
    :return: Distance 各结点之间的距离矩阵
    """
    # 将矩阵 [n , m ] 拓展为 [n , 1 , m ]
    expand_ = data[:, np.newaxis, :]
    # 将拓展的矩阵进行延申使得 n个[n,m] 即 [n,n,m]
    repeat1 = np.repeat(expand_, data.shape[0], axis=1)
    repeat2 = np.swapaxes(repeat1, 0, 1)
    # 求2范数
    D_pre = np.linalg.norm(repeat1 - repeat2, ord=2, axis=-1, keepdims=True)
    D_new = D_pre.squeeze(-1)
    return D_new


def get_matrix_B(D):
    """计算内积矩阵B"""
    # 计算之前得先断言为nxn矩阵
    assert D.shape[0] == D.shape[1]
    # 计算 dist_ij^2
    DD = np.square(D)
    # 计算行和
    sum_ = np.sum(DD, axis=1) / D.shape[0]
    Di = np.repeat(sum_[:, np.newaxis], D.shape[0], axis=1)
    Dj = np.repeat(sum_[np.newaxis, :], D.shape[0], axis=0)
    Dij = np.sum(DD) / ((D.shape[0]) ** 2) * np.ones([D.shape[0], D.shape[0]])
    B = (Di + Dj - DD - Dij) / 2
    return B


def mult_dim_scaling(data, n=2):
    D = get_distance_matrix(data)
    B = get_matrix_B(D)
    # 计算特征值、特征向量
    B_value, B_vector = np.linalg.eigh(B)
    # 对特征向量进行降序
    Be_sort = np.argsort(-B_value)
    # 降序排列的特征值
    B_value = B_value[Be_sort]
    # 降序排列的特征值对应的特征向量
    B_vector = B_vector[:, Be_sort]
    # 对角矩阵 - 特征值
    Bez = np.diag(B_value[0:n])
    # 选取后的特征向量
    Bvz = B_vector[:, 0:n]
    Z = np.dot(np.sqrt(Bez), Bvz.T).T
    return Z


class local_linear_embedding():
    def __init__(self, data, k_neigh, dim):
        self.data = data
        self.k_neigh = k_neigh
        self.dim = dim

    def fit_trans(self):
        """
        1. 对数据进行矩阵化（传入的参数一开始就是矩阵)
        2. for 1...m: 找到xi 的 k个近邻点,求 w_ij
        这时候得到一个W矩阵
        3. 求M矩阵：M = (I-W)(I-W)^T
        4。 求特征值
        5. 选d个特征向量，组成的就是低维的数据
        :return:
        """
        K_matrix = self.get_k_neighbors() # 得到近邻矩阵
        z_conv = np.cov(K_matrix, rowvar=0)

    def get_distance_matrix(self):
        """
        求距离矩阵
        :param data: n x m 矩阵数据
        :return: Distance 各结点之间的距离矩阵
        """
        data = self.data
        # 将矩阵 [n , m ] 拓展为 [n , 1 , m ]
        expand_ = data[:, np.newaxis, :]
        # 将拓展的矩阵进行延申使得 n个[n,m] 即 [n,n,m]
        repeat1 = np.repeat(expand_, data.shape[0], axis=1)
        repeat2 = np.swapaxes(repeat1, 0, 1)
        # 求2范数
        D_pre = np.linalg.norm(repeat1 - repeat2, ord=2, axis=-1, keepdims=True)
        D_new = D_pre.squeeze(-1)
        return D_new

    def get_k_neighbors(self):
        """
        找k近邻
        :param n_neigh:
        :return: nxk的近邻矩阵 其余部分为0
        """
        k_neigh = self.k_neigh
        # 距离矩阵
        D_matrix = self.get_distance_matrix()
        # 定义一个矩阵，存储近邻点序号
        n = D_matrix.shape[0]
        K_matrix = np.zeros((n, k_neigh))  # nxd
        # 遍历每一个点
        for i in range(n):
            index_ = np.argsort(D_matrix[i])
            K_matrix[i] = index_[1:k_neigh + 1]

        return K_matrix


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
    pca_data = np.array(lowDDataMat)

    mds_data = mult_dim_scaling(points, 2)
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(points)

    # mds = MDS(n_components=2)
    # X_mds = mds.fit_transform(points)

    plt.subplot(121)
    plt.scatter(pca_data[:, 0], pca_data[:, 1], marker='o')
    plt.subplot(122)
    plt.scatter(mds_data[:, 0], mds_data[:, 1], marker='o')
    plt.show()


def draw_o3d(file):
    """
    visualization of point clouds.
    :param file: ply
    :return:
    """
    pcd = o3d.io.read_point_cloud(file)
    o3d.visualization.draw_geometries([pcd], width=1000, height=1008)



if __name__ == '__main__':
    # dim_reduct_plot(OBJ_file)
    mesh = meshio.read(OBJ_file)
    points = mesh.points
    arrays = LLe = LLE(points,5,2)
    print(arrays)