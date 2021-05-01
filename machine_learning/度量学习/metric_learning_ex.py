# 距离测度学习的目的即为了衡量样本之间的相近程度，而这也正是模式识别的核心问题之一。
# 大量的机器学习方法，比如K近邻、支持向量机、径向基函数网络等分类方法以及K-means聚类方法，还有一些基于图的方法，其性能好坏都主要有样本之间的相似度量方法的选择决定。


# large margin nearest neighbor
from metric_learn import LMNN
import numpy as np

X = np.array([[0., 0., 1.], [0., 0., 2.], [1.,0.,0.], [2.,0.,0.], [2.,2.,2.], [2.,5.,4.]])
Y = np.array([1, 1, 2, 2, 0, 0])

lmnn = LMNN()