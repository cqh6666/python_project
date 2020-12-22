import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score


### EM算法流程
## 1. 初始化分布参数
##    while
##     E: 根据参数初始化值或上一次迭代的模型参数来计算出隐型变量的后验概率
##     M: 将似然函数最大化以获得新的参数值

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)

    plt.show()

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)

    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * 0.5) * (7 - i) * flip)

    # plt.show()


def sns_test():
    with sns.axes_style("darkgrid"):
        plt.subplot(211)
        sinplot()

    plt.subplot(212)
    sinplot(-1)
    # plt.show()


def GMM_test():
    X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=0)
    X = X[:, ::-1]  # 交换列是为了方便画图

    # 高斯混合模型
    gmm = GaussianMixture(n_components=4).fit(X)
    labels = gmm.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    plt.show()


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score


class GMM:
    def __init__(self,Data,K,weights = None,means = None,covars = None):
        """
        这是GMM（高斯混合模型）类的构造函数
        :param Data: 训练数据
        :param K: 高斯分布的个数
        :param weigths: 每个高斯分布的初始概率（权重）
        :param means: 高斯分布的均值向量
        :param covars: 高斯分布的协方差矩阵集合
        """
        self.Data = Data
        self.K = K
        if weights is not None:
            self.weights = weights
        else:
            self.weights  = np.random.rand(self.K)
            self.weights /= np.sum(self.weights)        # 归一化
        col = np.shape(self.Data)[1]
        if means is not None:
            self.means = means
        else:
            self.means = []
            for i in range(self.K):
                mean = np.random.rand(col)
                #mean = mean / np.sum(mean)        # 归一化
                self.means.append(mean)
        if covars is not None:
            self.covars = covars
        else:
            self.covars  = []
            for i in range(self.K):
                cov = np.random.rand(col,col)
                #cov = cov / np.sum(cov)                    # 归一化
                self.covars.append(cov)                     # cov是np.array,但是self.covars是list

    def Gaussian(self,x,mean,cov):
        """
        这是自定义的高斯分布概率密度函数
        :param x: 输入数据
        :param mean: 均值数组
        :param cov: 协方差矩阵
        :return: x的概率
        """
        dim = np.shape(cov)[0]
        # cov的行列式为零时的措施
        covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        xdiff = (x - mean).reshape((1,dim))
        # 概率密度
        prob = 1.0/(np.power(np.power(2*np.pi,dim)*np.abs(covdet),0.5))*\
               np.exp(-0.5*xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob

    def GMM_EM(self):
        """
        这是利用EM算法进行优化GMM参数的函数
        :return: 返回各组数据的属于每个分类的概率
        """
        loglikelyhood = 0
        oldloglikelyhood = 1
        len,dim = np.shape(self.Data)
        # gamma表示第n个样本属于第k个混合高斯的概率
        gammas = [np.zeros(self.K) for i in range(len)]
        while np.abs(loglikelyhood-oldloglikelyhood) > 0.00000001:
            oldloglikelyhood = loglikelyhood
            # E-step
            for n in range(len):
                # respons是GMM的EM算法中的权重w，即后验概率
                respons = [self.weights[k] * self.Gaussian(self.Data[n], self.means[k], self.covars[k])
                                                    for k in range(self.K)]
                respons = np.array(respons)
                sum_respons = np.sum(respons)
                gammas[n] = respons/sum_respons
            # M-step
            for k in range(self.K):
                #nk表示N个样本中有多少属于第k个高斯
                nk = np.sum([gammas[n][k] for n in range(len)])
                # 更新每个高斯分布的概率
                self.weights[k] = 1.0 * nk / len
                # 更新高斯分布的均值
                self.means[k] = (1.0/nk) * np.sum([gammas[n][k] * self.Data[n] for n in range(len)], axis=0)
                xdiffs = self.Data - self.means[k]
                # 更新高斯分布的协方差矩阵
                self.covars[k] = (1.0/nk)*np.sum([gammas[n][k]*xdiffs[n].reshape((dim,1)).dot(xdiffs[n].reshape((1,dim))) for n in range(len)],axis=0)
            loglikelyhood = []
            for n in range(len):
                tmp = [np.sum(self.weights[k]*self.Gaussian(self.Data[n],self.means[k],self.covars[k])) for k in range(self.K)]
                tmp = np.log(np.array(tmp))
                loglikelyhood.append(list(tmp))
            loglikelyhood = np.sum(loglikelyhood)
        for i in range(len):
            gammas[i] = gammas[i]/np.sum(gammas[i])
        self.posibility = gammas
        self.prediction = [np.argmax(gammas[i]) for i in range(len)]

def run_main():
    """
        这是主函数
    """
    # 导入数据集
    digits = load_digits()
    label = digits.target
    data = digits.data
    print("MNIST数据集的标签：\n",label)

    pca = PCA(0.9)
    data = pca.fit_transform(data)

    # GMM模型
    K = 10
    gmm = GMM(data,K)
    gmm.GMM_EM()
    y_pre = gmm.prediction
    print("GMM预测结果：\n",y_pre)
    print("GMM正确率为：\n",accuracy_score(label,y_pre))





def GMM_mnist():
    digits = load_digits()
    data = digits.data
    # plot_digits(data)
    pca = PCA(0.99,whiten=True)
    data = pca.fit_transform(data)

    gmm = GaussianMixture(n_components=10,covariance_type='diag').fit(data)
    labels = gmm.predict(data)

    print(digits.target, labels)

    prec = accuracy_score(digits.target, labels)
    print(prec)


GMM_mnist()