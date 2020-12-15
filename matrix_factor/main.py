import numpy as np
import matplotlib.pyplot as plt

class basic_mat_fac():
    """
        输入矩阵 R - nxm , alpha , beta , epochs
        初始化 P-nxk，Q-kxm

        伪代码:
        迭代更新(epochs)
            Loss = 0
            for i - [0,n)
                for j - [0,m)
                    loss_ij = r_ij - sum_{1-k}( p_ik * q_kj )
                    if r_ij>0:
                        LOSS = LOSS + (loss_ij ^ 2)
                        for k - [0,k)
                            p_ik = p_ik + alpha(2 * loss_ij * q_kj - beta * p_ik)
                            q_jk = q_kj + alpha(2 * loss_ij * p_ik - beta * q_kj)
                            LOSS = LOSS + ( beta/2 ) sum_{1-k}( p_ik ^ 2 + q_kj ^ 2 )

            if LOSS<0.001:
                break;

        predictR = P * Q
        输出 P,Q,predictR

        :return:P,Q,predictR
    """

    def __init__(self, R, P, Q):
        self.R = R
        self.P = P
        self.Q = Q

    def forward(self, K=5, epochs=10000, alpha=0.002, beta=0.02):
        R = self.R  # n x m
        P = self.P  # n x k
        Q = self.Q  # k x m

        loss_result = [] # 每一次迭代的损失值
        # 迭代epochs次
        for step in range(epochs):
            LOSS = 0  # 每一次迭代的总LOSS值
            for i in range(len(R)):  # 行数
                for j in range(len(R[i])):  # 列数
                    loss_ij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    if R[i][j] > 0:  # 矩阵内的数字有效
                        for k in range(K):
                            P[i][k] = P[i][k] + alpha * (2 * loss_ij * Q[k][j] - beta * P[i][k])
                            Q[k][j] = Q[k][j] + alpha * (2 * loss_ij * P[i][k] - beta * Q[k][j])
                            # 正则项的损失函数
                            LOSS = LOSS + np.power(loss_ij, 2) + (beta / 2) * (
                                    np.power(P[i][k], 2) + np.power(Q[k][j], 2))

            loss_result.append(LOSS)
            if step and (step) % 500 ==0:
                print("step:{}/{} | loss:{}".format(step,epochs,LOSS))
            if LOSS < 1e-3:
                break;

        predictR = np.dot(P,Q)
        return P,Q,predictR,loss_result


class prob_mat_fac():
    def __init__(self):
        pass

    def forward(self):
        pass


def main():
    R = [
        [5,0,0,3,2],
        [0,5,3,0,5],
        [5,3,0,0,0],
        [3,0,5,4,0]
    ]
    R = np.array(R)
    R_n = len(R)
    R_m = len(R[0])
    K = 3
    P = np.random.rand(R_n,K)
    Q = np.random.rand(K,R_m)

    mf = basic_mat_fac(R,P,Q)
    P_new,Q_new,mR,loss_result = mf.forward(K=K)
    mR = mR.astype(int)

    print("R:\n",R)
    print("mR:\n",mR)
    predictR = np.random.rand(R_n,R_m)
    for i in range(R_n):
        for j in range(R_m):
            if R[i][j]==0:
                predictR[i][j] = mR[i][j]
            else:
                predictR[i][j] = R[i][j]

    print("so the predictR is :\n",predictR)
    plot_loss(loss_result)


def plot_loss(loss_list):
    """
    输入列表，输出图像
    :param loss_list:损失值列表
    :return:
    """
    plt.plot(range(len(loss_list)),loss_list)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()

main()