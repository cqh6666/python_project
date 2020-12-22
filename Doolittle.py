# 矩阵的直接三角分解法

# 系数矩阵的三个斜边
# a,b,c

#
L = [0, 0]
D = [0]
U = [0]


def calc_doolittle(a, b, c, N):
    U.append(b[1])
    # [0,N]
    for i in range(1, N):
        D.append(c[i])
    for i in range(2, N + 1):
        L.append(a[i] / U[i - 1])
        U.append(b[i] - L[i] * c[i - 1])


# def zhui(f):
#     Y = []
#
#
# A = [0,0,-1,-1]
# B = [0,4,4,4]
# C = [0,-1,-1]
#
# calc_doolittle(A,B,C,3)
# print(L,D,U)


# 追赶法
import numpy as np

A = [[3.00000, 2.00000, 0.00000, 0.00000], [-1.00000, 3.00000, 2.00000, 0.00000], [0.00000, -1.00000, 3.00000, 2.00000],
     [0.00000, 0.00000, -1.00000, 3.00000]]
b = [[7.00000, 11.00000, 15.00000, 9.00000]]


def get_base(A):  # 获得一个基，在上面修改得到答案
    base = list(np.zeros((len(A), len(A))))
    D = []
    for i in base:
        D.append(list(i))
    return D


def get_gamma(A):  # 根据第一行公式γi=di
    base = get_base(A)
    for i in range(1, len(A)):
        base[i][i - 1] = A[i][i - 1]
    return base


def get_raw1(A, base):  # 根据第一行公式
    base[0][0] = A[0][0]
    base[0][1] = A[0][1] / base[0][0]
    return base


def get_other_raws(A, base, i):  # 递推后面几行
    base[i][i] = A[i][i] - A[i][i - 1] * base[i - 1][i]
    base[i][i + 1] = A[i][i + 1] / base[i][i]
    return base


def get_final_raw(A, base):  # 最后一行少一个β，另外求解，也可以和上面放在一起
    base[-1][-1] = A[-1][-1] - A[-1][-2] * base[-2][-1]
    return base


def get_all(A):  # 得到一个L和U并在一起的矩阵，由于U的主对角线为1，因此可以放在一起
    base = get_base(A)
    base = get_gamma(A)
    base = get_raw1(A, base)
    for i in range(1, len(A) - 1):
        get_other_raws(A, base, i)
    get_final_raw(A, base)
    return base


def get_lower(A):  # 获得L
    for i in A:
        for j in i:
            if i.index(j) > A.index(i):
                A[A.index(i)][i.index(j)] = 0
    return A


def get_upper(A):  # 获得U
    for i in A:
        for j in i:
            if i.index(j) < A.index(i):
                A[A.index(i)][i.index(j)] = 0.0
            elif i.index(j) == A.index(i):
                A[A.index(i)][i.index(j)] = 1.0
    return A


def f(x):
    return np.sqrt(x)


def fp(x):
    return 1 / (2 * np.sqrt(x))


A = [[4, 1],
     [1, 4]]

b1 = 3 * (f(102) - f(100)) - fp(100)
b2 = 3 * (f(103) - f(101)) - fp(103)
b = np.transpose([b1, b2])
X = np.linalg.solve(A, b)
print("b:\n", b)
print("方程组的解：\n", X);


def f519(x):
    y = np.power(x, 3) * np.exp(np.power(x, 2)) - np.sin(x)
    return y


def f5414(x, h):
    return 1 / (2 * h) * (-3 * f519(x) + 4 * f519(x + h) - f519(x + 2 * h))


def f5415(x, h):
    return 1 / (2 * h) * (-f519(x - h) + f519(x + h))


def s(x):
    return{
        'a':10,
        'b':20,
    }.get(x,99)

print(s('b'))
