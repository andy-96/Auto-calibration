import time, cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from functools import partial
from multiprocessing import Pool, cpu_count
from utils import nCr, homogenize_ncoord
from svd_ransac import fit_line, evaluate_line, generate_lines
from svd_estimate import display_lines, get_terminals

def estimate_line(kernel_pts, pts_homo, threshold, return_idx=True):
    if isinstance(kernel_pts, np.ndarray):
        kernel = kernel_pts.copy()
    else:
        kernel = np.vstack(kernel_pts)
    # fit a line to the kernel
    line = fit_line(kernel)
    # evaluate the line
    correctness, idx_fit = evaluate_line(line, pts_homo, threshold)
    if return_idx:
        return idx_fit
    else:
        return len(idx_fit)

def ransac_line_paral(pts, num_iter=24, kernel_size=2, threshold=0.5):
    # create homogeneous coordinates of points
    assert pts.ndim == 2, f"[estimate_line_paral]: pts has invalid dimension {pts.ndim}!"
    pts_homo = homogenize_ncoord(pts, ncoord=2)
    num_pts = pts_homo.shape[0]
    # check number of points
    if num_pts < 2:
        print(f"[estimate_line_paral]: not enough points {num_pts} < 2!")
        return None, 0, None, None
    # check kernel size
    if kernel_size > num_pts:
        print(f"[estimate_line_paral]: kernel size {kernel_size} > number of points {num_pts}! Set kernel size = {num_pts}!")
        kernel_size = num_pts
    elif kernel_size < 2:
        print(f"[estimate_line_paral]: kernel size {kernel_size} < minimal requirement 2! Set kernel size = 2!")
        kernel_size = 2
    # list all possible combinations
    if num_iter >= nCr(num_pts, kernel_size): # number of iterations are more than that of combinations
        cb = list(combinations(pts_homo, kernel_size))
    else:
        cb = list()
        for it in range(num_iter):
            cb.append(pts_homo[np.random.choice(num_pts, kernel_size, replace=False), :])
    # estimate lines parallelly
    pool = Pool(cpu_count())
    tmp_func = partial(estimate_line, pts_homo=pts_homo, threshold=threshold, return_idx=False)
    correctness_list = pool.map(tmp_func, cb)
    assert len(correctness_list) == len(cb), f"[estimate_line_paral]: Number of combinations {len(cb)} != Number of correctness {len(correctness_list)}!"
    pool.close()
    # find the line with highest correctness
    cb_idx = correctness_list.index(max(correctness_list))
    best_idx_fit = estimate_line(cb[cb_idx], pts_homo, threshold)
    best_line = fit_line(pts_homo[best_idx_fit, :])
    # check consistency
    best_correctness = correctness_list[cb_idx]
    assert len(best_idx_fit) == best_correctness, f"[estimate_line_paral]: Correctness {best_correctness} != Number of support points {len(best_idx_fit)}!"

    return best_line, best_correctness, best_idx_fit, np.vstack(cb[cb_idx])
    
def generate_lines_paral(pts, num_lines, dist_thresh=1, correctness_thresh=8, kernel_size=2, faster=True, keep_trace=True):
    # create homogeneous coordinates of points
    assert pts.ndim == 2, f"[estimate_line_paral]: pts has invalid dimension {pts.ndim}!"
    pts_homo = homogenize_ncoord(pts, ncoord=2)
    # fit estimate lines
    lines = list() # list of ndarray(3, )
    terminals = list() # list of ndarray(2, 2)
    pts_target = pts_homo.copy() # candidate points (homogeneous) for estimating ONE line 
    for i in tqdm(range(num_lines), desc="Generating Line", leave=keep_trace, position=0):
        num_targets = pts_target.shape[0]
        # check number of points
        if num_targets < 2: continue
        if faster:
            p = 0.999
            s = kernel_size
            w = 1./(num_lines-i)**2 # probability of inliers
            max_iter = int(np.log2(1-p)/np.log2(1-w**s+1e-8))
        else:
            max_iter = nCr(num_targets, kernel_size)
        if max_iter < 1: max_iter = 1
        line, correctness, idx_fit, _ = ransac_line_paral(pts_target, num_iter=max_iter, kernel_size=kernel_size, threshold=dist_thresh)
        # drop lines with low correctness
        if correctness < correctness_thresh: continue
        # no more lines can be estimated
        if idx_fit is None: break
        # calculate line terminal pairs from the support points
        pts_fit = pts_target[idx_fit, :-1] # ?x2
        terminals.append(get_terminals(pts_fit))
        # store the line
        lines.append(line)
        # remove points already fitted with lines
        pts_target = np.delete(pts_target, idx_fit, axis=0)
    return lines, terminals

if __name__ == '__main__':
    # img = cv2.imread('improve/6-6/6_6.jpg')
    img = np.ones((1440, 1920, 3))*255
    with open(f'./frame012553-0_inner_polygon.txt', 'r') as f:
        data = f.readlines()
    pts = []
    for d in data:
        d = d.strip()
        d_list = d.split(' ')
        pts.append((float(d_list[0]), float(d_list[1])))
    pts = np.array(pts, dtype=np.float)
    tic1 = time.time()
    lines1, terminals1 = generate_lines_paral(pts, 8, dist_thresh=0.5, correctness_thresh=12, kernel_size=2, faster=True)
    toc1 = time.time()
    print(f"Parallel Time = {round((toc1-tic1)*1000, 1)}ms")
    tic2 = time.time()
    lines2, terminals2 = generate_lines(pts, 8, dist_thresh=0.5, correctness_thresh=12, kernel_size=2, faster=True)
    toc2 = time.time()
    print(f"Sequential Time = {round((toc2-tic2)*1000, 1)}ms")
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img[:, :, ::-1])
    fig, axes[0] = display_lines(img, lines1, terminals1, fig=fig, ax=axes[0], line_format='-g')
    axes[0].set_title(f"Parallel Time = {round((toc1-tic1)*1000, 1)}ms")
    axes[1].imshow(img[:, :, ::-1])
    fig, axes[1] = display_lines(img, lines2, terminals2, fig=fig, ax=axes[1], line_format='-g')
    axes[1].set_title(f"Sequential Time = {round((toc2-tic2)*1000, 1)}ms")
    plt.show()
    