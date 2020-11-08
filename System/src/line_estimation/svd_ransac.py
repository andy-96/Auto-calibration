import cv2
# import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from svd_estimate import display_lines, get_round_intersections, get_terminals
from utils import nCr
from itertools import combinations
import time

def find_neighbors(pts_homo, seed_homo, k=2):
    assert pts_homo.shape[1] == 3, "In find_neighbors: pts_homo has wrong shape!"
    assert seed_homo.shape == (1, 3), "In find_neighbors: seed_homo has wrong shape!"
    dist = np.sqrt(np.sum((pts_homo-seed_homo)**2, axis=1))
    knn_idx = np.argsort(dist)[:k]
    knn_pts = pts_homo[knn_idx]
    return knn_pts # k x 3

# normal: ndarray (3, ), [a, b, c], ax + by + c = 0
def fit_line(pts): # pts: ndarray (N, 3) or (N, 2)
    return get_components(pts, [-1]).flatten()

def get_components(pts, idx):
    num_pts = pts.shape[0]
    if pts.shape == (num_pts, 2):
        pts_homo = np.hstack((pts, np.ones((num_pts, 1))))
    elif pts.shape == (num_pts, 3):
        pts_homo = pts.copy()
    else:
        print(f"[get_components]: 'pts' has wrong shape: {pts.shape}!")
    if isinstance(idx, int) or isinstance(idx, list):
        pass
    elif isinstance(idx, np.ndarray):
        assert idx.ndim == 1, f"[get_components]: invalid dimension of 'idx' array {idx.ndim}"
    elif isinstance(idx, float):
        assert idx == int(idx), f"[get_components]: 'idx' has decimal number {idx}"
    else:
        print(f"[get_components]: invalid type of 'idx' {type(idx)}")
    
    u, s, vt = np.linalg.svd(pts_homo)
    return vt[idx, :] # (?, 3) or (3, )

# correctness: int
# idx[0]: ndarray (N, )
def evaluate_line(normal, pts_homo, threshold):
    assert normal.shape == (3, ), "[evaluate_line]: normal has wrong shape!"
    assert pts_homo.shape[1] == 3, "[evaluate_line]: pts_homo has wrong shape!"
    # distance from points to the line
    norm = np.sqrt(np.sum(normal[:-1]**2)) # \sqrt (a^2 + b^2)
    dist = np.abs(pts_homo @ normal.reshape((-1, 1))) / norm # (ax + by + c) / norm
    # find points close enough to the line
    idx = np.where(dist < threshold)
    pts_fit = pts_homo[idx[0], :-1]
    # correctness = # of close points
    correctness = len(idx[0])
    return correctness, idx[0]

def ransac_line(pts_homo, num_iter=24, kernel_size=4, threshold=5):
    assert pts_homo.ndim == 2 and pts_homo.shape[1] == 3, f"[ransac_line]: pts_homo has wrong shape {pts_homo.shape}!"
    num_pts = pts_homo.shape[0]
    # check number of points
    if num_pts < 2:
        print(f"[ransac_line]: not enough points {num_pts} < 2!")
        return None, 0, None, None
    # check kernel size
    if kernel_size > num_pts:
        print(f"[ransac_line]: kernel size {kernel_size} > number of points {num_pts}! Set kernel size = {num_pts}!")
        kernel_size = num_pts
    elif kernel_size < 2:
        print(f"[ransac_line]: kernel size {kernel_size} < minimal requirement 2! Set kernel size = 2!")
        kernel_size = 2
    ## sample randomly or go through all possible combinations
    is_sampled = True
    if num_iter >= nCr(pts_homo.shape[0], kernel_size): # number of iterations are more than the number of combinations
        cb = list(combinations(pts_homo, kernel_size))
        num_iter = len(cb)
        is_sampled = False
    ## recorder
    best_kernel_pts = None # points used to estimate the line
    best_correctness = 0
    best_line = None # line with highest correctness
    best_idx_fit = None # points support the best line
    ## RANSAC
    for i in range(num_iter):
        # sample/generate kernel
        if is_sampled:
            kernel_pts = pts_homo[np.random.choice(pts_homo.shape[0], kernel_size, replace=False), :]
        else:
            kernel_pts = np.vstack(cb[i])
        # fit line
        line = fit_line(kernel_pts) # line normal parameters (a, b, c)
        # evaluate line
        correctness, idx_fit = evaluate_line(line, pts_homo, threshold)
        # update best records
        if correctness > best_correctness:
            best_kernel_pts = kernel_pts
            best_correctness = correctness
            best_line = line
            best_idx_fit = idx_fit
    # with open('before.txt', 'a') as f:
    #     row = ' '.join(str(p) for p in best_line)
    #     f.write(row + '\n')
    # estimate the best line using support points (SVD)
    best_line = fit_line(pts_homo[best_idx_fit, :])
    # with open('after.txt', 'a') as f:
    #     row = ' '.join(str(p) for p in best_line)
    #     f.write(row + '\n')
    # exit()
    return best_line, best_correctness, best_idx_fit, best_kernel_pts

r''' 
    params:
        pts                : 2d (homogeneous) coordinates of points - ndarray (N, 2) or (N, 3)
        num_lines          : number of fitting lines desired
        dist_thresh        : point to line distance upper bound
        correctness_thresh : minimum number of points supporting the fitting line to keep the line
        kernel_size        : model size, number of points to sample for proposing one line
        faster             : use less iterations with 99.9% probability guanrantee of sampling all inliner points 
    return: 
        lines    : list of line vectors - [ndarray (3, )]
        terminals: list of end points for line segments above - [ndarray (2, 2)]
                   [array([
                           [x1, y1],
                           [x2, y2]
                       ])]
'''
def generate_lines(pts, num_lines, dist_thresh=1, correctness_thresh=4, kernel_size=2, faster=True, keep_trace=True):
    num_pts = pts.shape[0]
    if pts.shape == (num_pts, 2):
        pts_homo = np.hstack((pts, np.ones((num_pts, 1))))
    elif pts.shape == (num_pts, 3):
        pts_homo = pts.copy()
    else:
        print(f"[generate_lines]: 'pts' has wrong shape: {pts.shape}!")
        exit()

    # fit line by ransac
    lines = list() # list of ndarray(3, )
    terminals = list() # list of ndarray(2, 2)
    pts_target = pts_homo.copy()
    for i in tqdm(range(num_lines), desc="Generating Line", leave=keep_trace, position=0):
        # fit line
        num_target = pts_target.shape[0]
        # check number of points
        if num_target < 2: continue
        if faster:
            p = 0.999
            s = kernel_size
            # epsilon = 1. - 0.1/(num_lines-i) # probability of outliers
            # max_iter = int(np.log2(1-p)/np.log2(1-(1-epsilon)**s+1e-8))
            w = 1./(num_lines-i)**2 # probability of inliers
            max_iter = int(np.log2(1-p)/np.log2(1-w**s+1e-8))
        else:
            max_iter = nCr(num_target, kernel_size)
        if max_iter < 1: max_iter = 1
        line, correctness, idx_fit, _ = ransac_line(pts_target, num_iter=max_iter, kernel_size=kernel_size, threshold=dist_thresh) # threshold: point to line distance upper bound
        # filter low correctness lines
        if correctness < correctness_thresh: # at least 'correctness_thresh' number of points fit to a line
            continue
        if idx_fit is None: # no more lines fit
            break
        pts_fit = pts_target[idx_fit, :-1] # ?x2
        terminals.append(get_terminals(pts_fit))
        # store line and terminal points
        lines.append(line)
        # remove points already fitted with lines
        pts_target = np.delete(pts_target, idx_fit, axis=0)
        # print correctness
        # tqdm.write(f"correctness = {correctness}")
    return lines, terminals

if __name__ == "__main__":
    # image preprocessing
    img = np.array(plt.imread('../../Traffic_sign_simulation/Results/Traffic_sign(image1).jpg')) # RGB
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # gray[gray < 200] = 0

    # # detect interest points
    # pts = cv2.goodFeaturesToTrack(gray, 800, 0.01, 0.5).reshape((-1, 2)) # number of points, quality threshold, points' distance lower bound
    # num_pts = pts.shape[0]
    # print(f"Number of interest points: {num_pts}")
    ### load points ###
    txt_name = 'outer_polygon'
    with open(f'./txt/{txt_name}.txt', 'r') as f:
        data = f.readlines()
    pts = []
    for d in data:
        d = d.strip()
        d_list = d.split(' ')
        pts.append((float(d_list[0]), float(d_list[1])))
    pts = np.array(pts, dtype=np.float)
    num_pts = pts.shape[0]
    print(f"Number of interest points: {num_pts}")

    # # # major and minor axes
    # pc = list(get_components(pts, [-1, -2]))
    # fig, ax = display_lines(img, pc)

    # generate lines
    # lines, domains, ranges = generate_lines(pts, 8, dist_thresh=0.5, correctness_thresh=num_pts/20)
    lines, terminals = generate_lines(pts, 8, dist_thresh=0.5, correctness_thresh=num_pts/20)

    # intersections
    # intersections = get_round_intersections(lines)
    # np.savetxt(f'./intersections/{txt_name}.txt', intersections, fmt='%1.3f')
    
    # plt.imsave('binary_input.png', gray, cmap=plt.get_cmap('gray'))
    # display
    # fig, ax = display_lines(img, lines, domains, ranges, fig, ax)
    # fig, ax = display_lines(img, lines, domains, ranges)
    fig, ax = display_lines(img, lines, terminals)
    ax.imshow(img)
    ax.scatter(pts[:, 0], pts[:, 1], c='g', s=1, zorder=1)
    ax.set_title(f'Points detected = {num_pts}\n Lines fit = {len(lines)}')
    # fig.savefig('traffic_sign0_line.png')
    plt.show()


