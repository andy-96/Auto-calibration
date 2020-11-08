import numpy as np
from math import tan, pi
from os import path, mkdir

def get_physical_pts(num_polygon=8, num_T=4):
    d = 925
    s = d * tan(((45/2)/180)*pi)
    inner_polygon_pts = np.array([
        [175, 637.5-s/2],
        [637.5-s/2, 175],
        [637.5+s/2, 175],
        [1100, 637.5-s/2],
        [1100, 637.5+s/2],
        [637.5+s/2, 1100],
        [637.5-s/2, 1100],
        [175, 637.5+s/2],
    ], dtype=np.float32)
    inner_T_pts = np.array([
        [492, 475],
        [540, 475],
        [540, 518],
        [492, 518],
    ], dtype=np.float32)
    origin = np.array([150, 150])
    scale_ratio = 15./25
    inner_polygon_physical = np.zeros((8, 3))
    inner_T_physical = np.zeros((4, 3))
    inner_polygon_physical[:, :2] = (inner_polygon_pts-origin)*scale_ratio
    inner_T_physical[:, :2] = (inner_T_pts-origin)*scale_ratio
    return inner_polygon_physical[:num_polygon, :], inner_T_physical[:num_T, :]

    # if not path.isdir(f'{root}/physical'):
    #     mkdir(f'{root}/physical')
    # np.savetxt(f'{root}/physical/inner_polygon.txt', (inner_polygon_pts-origin)*scale_ratio, fmt='%1.3f')
    # np.savetxt(f'{root}/physical/inner_T.txt', (inner_T_pts-origin)*scale_ratio, fmt='%1.3f')
    # print(f"inner_polygon.txt & inner_T.txt saved under {root}/physical")

if __name__ == '__main__':
    root = './improve'
    inner_polygon, inner_T = get_physical_pts(num_polygon=8, num_T=4)
    np.savetxt(f'{root}/inner_polygon.txt', inner_polygon, fmt='%1.3f')
    np.savetxt(f'{root}/inner_T.txt', inner_T, fmt='%1.3f')

