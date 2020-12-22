import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


objPath = "./dataset/xatlas-web/models"

FilePath = os.path.join(objPath, "matteonormb.obj")

def load_data(FILE_PATH):
    with open(FILE_PATH) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
    # points原本为列表，需要转变为矩阵，方便处理
    points = np.array(points)
    return points

points = load_data(FilePath)

x = points[:][0]
y = points[:][1]
z = points[:][2]

print(x,y,z)




