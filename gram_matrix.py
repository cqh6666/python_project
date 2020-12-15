import numpy as np

x = [0,
     0.5,
     1,
     1.5,
     2,
     2.5,
     3,
     3.5,
     4]

y = [4,
     2.927,
     2.47,
     2.393,
     2.54,
     2.829,
     3.198,
     3.621,
     4.072]

datax_418 = [0,
             0.9,
             1.9,
             3,
             3.9,
             5
             ]
datay_418 = [0,
             10,
             30,
             50,
             80,
             110

             ]

datax_419 = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
datay_419 = [4, 6.41, 8.01, 8.79, 9.53, 9.86, 10.33, 10.42, 10.53, 10.61]

data_x_419 = [1/i for i in datax_419]
data_y_419 = [1/i for i in datay_419]
print("转换后的x,z",data_x_419,data_y_419)



def calc_vec(data_x):
    """fail0 为 第一个基对应的x，fail1为 第二个基对应的x， 主要修改 [ * for ... ]"""
    fai0 = [i for i in data_x];
    fai1 = [1 for i in data_x]
    return fai0, fai1


def gram_mat(fai0, fai1, y):
    """两个未知数的特例 a+bx 的 a和b"""
    m1 = np.dot(fai0, np.transpose(fai0))
    m2 = np.dot(fai1, np.transpose(fai0))
    m3 = np.dot(fai0, np.transpose(fai1))
    m4 = np.dot(fai1, np.transpose(fai1))
    gmat = [[m1, m2], [m3, m4]]
    y1 = np.dot(y, np.transpose(fai0))
    y2 = np.dot(y, np.transpose(fai1))
    ymat = [y1, y2]
    return gmat, ymat

# data_x_419 为 x值
fai0, fai1 = calc_vec(data_x_419)
print("fai0,fai1,y", fai0, fai1, data_y_419)
gmat, ymat = gram_mat(fai0, fai1, data_y_419)
print(gmat, ymat)

# 矩阵求解
a = np.linalg.solve(gmat, ymat)
print("the solve:", a)

# 平局误差 , data_y_419 为 y值
data_y_419 = np.array(data_y_419)
error1 = np.sum(data_y_419 ** 2)
error2 = a[0] * ymat[0] + a[1] * ymat[1]
kexi_error = error1 - error2
print("y的平方和,估计值和:", error1, error2)
print("平方误差：", kexi_error)
