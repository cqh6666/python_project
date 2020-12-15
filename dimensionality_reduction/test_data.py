import numpy as np

data = np.array([[1, 1, 2], [1, 3, 3]])
DD = np.square(data)
# 计算列和，共n列
sum_ = np.sum(DD, axis=1) / data.shape[0]
Di = np.repeat(sum_[:, np.newaxis], data.shape[0], axis=1)
Dj = np.repeat(sum_[np.newaxis, :], data.shape[0], axis=0)
Dij = np.sum(DD) / ((data.shape[0]) ** 2) * np.ones([data.shape[0], data.shape[0]])
B = (Di + Dj - DD - Dij) / 2
print(B)
