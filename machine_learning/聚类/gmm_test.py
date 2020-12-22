import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs


#产生实验数据
X, y_true = make_blobs(n_samples=1000, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] #交换列是为了方便画图

gmm = GaussianMixture(n_components=4,covariance_type='diag').fit(X)
labels = gmm.predict(X)
print(labels)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
plt.show()