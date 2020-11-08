import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# in case that the generated physical points are in the image uv coordinates (i.e. (vertical, horizontal)),
# swap them to get the normal xy coordinates (i.e. (horizontal, vertical))
def swap_coordinates(path, txt_name):
    # open file
    with open(path + '/' + txt_name, 'r') as f:
        data = f.readlines()
    # read data
    output_list = []
    for line in data:
        output_list.append([float(s) for s in line.strip().split()])
    output_array = np.array(output_list)
    # swap columns and save file
    np.savetxt(f"{path}/{txt_name}", output_array[:, [1, 0, 2]], fmt="%1.3f")
    tqdm.write(f"{txt_name} saved to {path}!")


# get list of row point arrays & column point arrays in homogeneous coordinates (i.e. (x, y, 1)): [ndarray (cnt*3), ndarray, ...]
def prepare_data(path, txt_name, row_cnt): # row_cnt: int, number of points in each row
    with open(path + '/' + txt_name, 'r') as f:
        data = f.readlines()

    assert len(data) % row_cnt == 0, f"{len(data)} points can't be distributed into rows with {row_cnt} equal points!"
    col_cnt = len(data) // row_cnt

    rows = [[] for i in range(row_cnt)]
    cols = [[] for i in range(col_cnt)]
    
    for idx, line in enumerate(data):
        point = [float(s) for s in line.strip().split()] + [float(1)]
        
        # distributes points to corresponding rows
        rows[idx // row_cnt].append(point)

        # distributes points to corresponding columns
        cols[idx % row_cnt].append(point)

    row_arrays = [np.array(r) for r in rows]
    col_arrays = [np.array(c) for c in cols]

    return row_arrays, col_arrays

# Assuming this set of points come from a line, find the terminal points of this line from this set of points
def get_terminals(pts):
    xmin_idx = np.argmin(pts[:, 0]); xmin = pts[xmin_idx, 0]
    xmax_idx = np.argmax(pts[:, 0]); xmax = pts[xmax_idx, 0]
    ymin_idx = np.argmin(pts[:, 1]); ymin = pts[ymin_idx, 1]
    ymax_idx = np.argmax(pts[:, 1]); ymax = pts[ymax_idx, 1]
    if (xmax - xmin) > (ymax - ymin): # range of x values is larger
        return np.array([
            pts[xmin_idx, :],
            pts[xmax_idx, :]
        ])
    else: # range of y values is larger
        return np.array([
            pts[ymin_idx, :],
            pts[ymax_idx, :]
        ])

# returns the normal vector ndarray(3, ) of a line, and the terminal points ndarray(2, 2)
def fit_line(pts): # pts: ndarray (N by 3) in homogeneous coordinates
    assert pts.shape[1] == 3, f"Array dimension {pts.shape} isn't valid!"
    u, s, vt = np.linalg.svd(pts)
    return vt[-1, :], get_terminals(pts)

def display_lines(img, normals, terminals=None, fig=None, ax=None, line_format='-k'): # img: ndarray; normals: [ndarray(3, ), ...], normal vectors of lines; terminals: [ndarray(2, 2), ...], terminal points of each line
    # fig & ax
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    # domains & ranges
    sdim = min(img.shape[0], img.shape[1])-1
    if terminals is None:
        ts = [
            np.array([
                [0, 0],
                [img.shape[0], img.shape[1]]
            ])
        ] * len(normals)
    else:
        assert len(normals) == len(terminals), f"The number of normal vectors doesn't match the number of terminal pairs: {len(normals)} v.s. {len(terminals)}!"
        ts = terminals.copy()
    # plot the lines
    for n, t in zip(normals, ts):
        if (abs(t[0, 0] - t[1, 0]) > abs(t[0, 1] - t[1, 1])): # small slope
            x = np.linspace(min(t[:, 0]), max(t[:, 0]), 100)
            y = -(n[0] * x + n[2]) / n[1]
        else: # large slope
            y = np.linspace(min(t[:, 1]), max(t[:, 1]), 100)
            x = -(n[1] * y + n[2]) / n[0]
        ax.plot(x, y, line_format, zorder=5, linewidth=1)
    return fig, ax
# def display_lines(img, normals, domains=None, ranges=None, fig=None, ax=None): # img: ndarray; normals: [ndarray(3, ), ...], normal vectors of lines; domains/ranges: [(min, max), ...], x/y boundaries of corresponding lines
#     # fig & ax
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.gca()
#     # domains & ranges
#     sdim = min(img.shape[0], img.shape[1])-1
#     if (domains is None) and (ranges is None):
#         ds = [(0, sdim)] * len(normals)
#         rs = [(0, sdim)] * len(normals)
#     elif domains is None:
#         assert len(normals) == len(ranges), f"[display_lines]: 'normals' and 'ranges' sizes don't match {len(normals)} v.s. {len(ranges)}!"
#         ds = [(0, sdim)] * len(normals)
#         rs = ranges.copy()
#     elif ranges is None:
#         assert len(normals) == len(domains), f"[display_lines]: 'normals' and 'domains' sizes don't match {len(normals)} v.s. {len(domains)}!"
#         ds = domains.copy()
#         rs = [(0, sdim)] * len(normals)
#     else:
#         assert len(normals) == len(domains) == len(ranges), f"The number of normal vectors doesn't match the number of domains or ranges: {len(normals)} v.s. {len(domains)} v.s. {len(ranges)}!"
#         ds = domains.copy()
#         rs = ranges.copy()
#     # plot the lines
#     for n, d, r in zip(normals, ds, rs):
#         if (d[1] - d[0]) > (r[1] - r[0]): # small slope
#             x = np.linspace(d[0], d[1], 100)
#             y = -(n[0] * x + n[2]) / n[1]
#         else: # large slope
#             y = np.linspace(r[0], r[1], 100)
#             x = -(n[1] * y + n[2]) / n[0]
#         ax.plot(x, y, '-k', zorder=5, linewidth=1)
#     return fig, ax

def get_pairwise_intersections(lines1, lines2): # lines1/lines2: [ndarray(3, ), ...], normal vectors of lines
    assert len(lines1) == len(lines2), f"Lines are not paired: {len(lines1)} v.s. {len(lines2)}" 
    intersections = [] # [ndarray(2, ), ...]
    for (n1, n2) in zip(lines1, lines2):
        A = np.array([
            [n1[0], n1[1]],
            [n2[0], n2[1]]
        ])
        v = np.array([
            [n1[2]],
            [n2[2]]
        ])
        intersections.append( (-np.linalg.inv(A) @ v).flatten() )
    return np.array(intersections)

def get_round_intersections(lines): # lines: [ndarray(3, ), ...], normal vectors of lines
    intersections = [] # [ndarray(2, ), ...]
    num = len(lines)
    for i in range(num):
        n1 = lines[i]
        for j in range(i+1, num):
            n2 = lines[j]
            A = np.array([
                [n1[0], n1[1]],
                [n2[0], n2[1]]
            ])
            v = np.array([
                [n1[2]],
                [n2[2]]
            ])
            intersections.append( (-np.linalg.inv(A) @ v).flatten() )
    return np.array(intersections)

# stores in row-then-column order
def get_grid_intersections(col_lines, row_lines): # col_lines/row_lines: [ndarray(3, ), ...], normal vectors of lines
    intersections = [] # [ndarray(2, ), ...]
    for n_r in row_lines:
        for n_c in col_lines:
            A = np.array([
                [n_r[0], n_r[1]],
                [n_c[0], n_c[1]]
            ])
            v = np.array([
                [n_r[2]],
                [n_c[2]]
            ])
            intersections.append( (-np.linalg.inv(A) @ v).flatten() )
    return np.array(intersections)


# if __name__ == "__main__":
#     txt_path = './chessboard/large1_point_swap/'
#     txt_names = sorted(os.listdir(txt_path))
#     if '.DS_Store' in txt_names:
#         txt_names.remove('.DS_Store')
#     for txt_name in tqdm(txt_names):
#         swap_coordinates(txt_path, txt_name)
if __name__ == "__main__":
    # hyper-parameters
    txt_path = './chessboard/large1_feature/'
    out_path = './chessboard/large1_feature_fitted/'
    name = 'large_cb9'
    txt_name = f'{name}.txt'
    row_cnt = 9
    img_path = './image/'
    img_name = f'{name}.jpg'
    # get row & column points
    row_arrays, col_arrays = prepare_data(txt_path, txt_name, row_cnt)
    # get image
    img = plt.imread(img_path+img_name)

    # fit lines
    row_lines = []
    row_terminals = []
    col_lines = []
    col_terminals = []
    for row_array in row_arrays:
        row_line, row_ts = fit_line(row_array)
        row_lines.append(row_line)
        row_terminals.append(row_ts)
    for col_array in col_arrays:
        col_line, col_ts = fit_line(col_array)
        col_lines.append(col_line)
        col_terminals.append(col_ts)

    # get intersections
    intersections = get_grid_intersections(col_lines, row_lines)

    # display lines
    fig, ax = display_lines(img, row_lines + col_lines, row_terminals + col_terminals)

    # display intersections
    ax.scatter(intersections[:, 0], intersections[:, 1], c='g', s=4, zorder=10)

    ax.set_title(f'Lines Fitted to Points: {txt_name}')
    # plt.show()
    # save image and the points
    fig.savefig(out_path + img_name)
    np.savetxt(out_path + txt_name, intersections, fmt='%1.3f')
    
