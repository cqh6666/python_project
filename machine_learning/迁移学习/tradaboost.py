import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


class TrAdaboost:
    def __init__(self, base_learner=DecisionTreeClassifier(), N=20):
        """

        :param base_learner:
        :param N:
        """
        self.base_learner = base_learner
        self.N = N
        self.beta_all = np.zeros([1, self.N])
        self.learners = []

    def fit(self, train_d, train_s, label_d, label_s):
        """
        the two labeled data set Td,Ts
        :param train_d:不同分布的training data
        :param label_d:
        :param train_s:相同分布的training data
        :param label_s:
        :return:
        """
        # 预处理数据集 将train_data 合并
        x_train = np.concatenate((train_d, train_s), axis=0)
        y_train = np.concatenate((label_d, label_s), axis=0)

        # 获取 train_data 的 数量
        num_diff = train_d.shape[0]
        num_same = train_s.shape[0]

        # 初始化权重
        weight_diff = np.ones([num_diff, 1]) / num_diff
        weight_same = np.ones([num_same, 1]) / num_same
        weights = np.concatenate((weight_diff, weight_same), axis=0)

        # diff 训练集 权重变量的底数
        beta = 1 / (1 + np.sqrt(2 * np.log(num_diff / self.N)))

        # 基学习器的预测值填入此数组 (k,N)
        y_pred = np.ones([num_diff + num_same, self.N])

        for i in range(self.N):
            weights = self.calculate_weight(weights)

            # call learner and save them, hypothesis the x_train result (h_t)
            self.base_learner.fit(x_train, y_train, sample_weight=weights[:, 0])
            self.learners.append(self.base_learner)  # 保存每次的学习器
            y_pred[:, i] = self.base_learner.predict(x_train)  # 1 - n+m

            # calculate the error of h_t on T_s
            error_rate = self.calculate_error_rate(y_pred[num_diff:, i], label_s[:], weights[num_diff:, 0])

            if error_rate > 0.5:
                error_rate = 0.5
            elif error_rate == 0:
                self.N = i
                print(self.N, ":we have to stop...")
                break

            print("[{0}/{1}] - error_rate:{2}".format(i, self.N - 1, error_rate))

            self.beta_all[0, i] = error_rate / (1 - error_rate)

            # Update the new weight vector
            for s in range(num_diff):
                weights[i] = weights[i] * np.power(beta, np.abs(y_pred[s, i] - label_d[s]))
            for t in range(num_same):
                weights[num_diff + t] = weights[num_diff + t] * np.power(self.beta_all[0, i], -np.abs(
                    y_pred[num_diff + t, i] - label_s[t]))

    def predict(self, test_x):
        """
        将 N/2 次后的预测器的结果与1/2 比较，整体来说>1/2 则 分类到1去
        :param test_x: 测试集合
        :return:
        """
        num_test = test_x.shape[0]

        result = np.zeros([num_test, self.N + 1])
        predict = np.zeros([num_test, 1])

        # 预测值放入result
        i = 0
        for base_learner in self.learners:
            result[:, i] = base_learner.predict(test_x)
            i = i + 1

        for t in range(num_test):
            start_n = (int)(np.ceil(self.N / 2))
            left = - np.sum(result[t, start_n:self.N] * np.log(self.beta_all[0, start_n:self.N]))
            right = - np.sum(0.5 * np.log(self.beta_all[0, start_n:self.N]))

            if left >= right:
                predict[t] = 1
            else:
                predict[t] = 0

        return predict

    def calculate_weight(self, weights):
        """
        计算新的weights
        :param weights: [n+m,]
        :return:
        """
        sum_weights = np.sum(weights)
        return weights / sum_weights

    def calculate_error_rate(self, y_pred, y_train, weights):
        sum_weights = np.sum(weights)
        return np.sum(weights * np.abs(y_pred - y_train) / sum_weights)


def run():
    tr = TrAdaboost()
    iris = load_iris()
    iris.target[iris.target > 0] = 1
    iris.target[iris.target == 0] = -1
    iris_x = iris.data
    iris_y = iris.target
    print(iris_x,iris_y)
    # 不同分布和同一分布
    train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=0.3, random_state=1)
    tr.fit(train_x, test_x, train_y, test_y)
    result = tr.predict(test_x)
    print(result,test_y)


if __name__ == "__main__":
    run()
