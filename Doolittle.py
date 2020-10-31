# 矩阵的直接三角分解法

# 系数矩阵的三个斜边
# a,b,c

#
L = [0,0]
D = [0]
U = [0]


def calc_doolittle(a,b,c,N):
    U.append(b[1])
    # [0,N]
    for i in range(1,N):
        D.append(c[i])
    for i in range(2,N+1):
        L.append(a[i]/U[i-1])
        U.append(b[i]-L[i]*c[i-1])


def zhui(f):
    Y = []


A = [0,0,-1,-1]
B = [0,4,4,4]
C = [0,-1,-1]

calc_doolittle(A,B,C,3)
print(L,D,U)