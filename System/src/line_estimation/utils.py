import operator as op
from functools import reduce
from math import pi, acos
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import glob
import os

def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def line_cos_angle(line1, line2): # line: (a, b, c) with ax + by + c = 0, ndarray(3, )
    (a1, b1), (a2, b2) = line1[:2]/np.linalg.norm(line1[:2]), line2[:2]/np.linalg.norm(line2[:2])
    # normalize
    cos_theta = a1 * a2 + b1 * b2
    return abs(cos_theta)

def dist(xi, xj, metric = 'euclidean'):
    diff = np.abs(xi - xj).astype(float)
    if metric == 'euclidean' or metric == 'l2':
        return np.sqrt(np.sum(diff**2, axis=diff.ndim-1))
    if metric == 'manhattan' or metric == 'l1':
        return np.sum(diff, axis=diff.ndim-1)
    if metric == 'diagonal' or metric == 'linf':
        return np.max(diff, axis=diff.ndim-1)
    raise NotImplementedError

def homogenize(x):
    if x.ndim < 1:
        print(f"[homogenize]Error: Invalid input {x}!")
    elif x.ndim == 1: # 1-d vector
        return np.hstack([x, 1]).astype(x.dtype)
    else: # higher dimensional tensor (or matrix)
        new_x = np.ones(x.shape[:-1] + (x.shape[-1]+1,))
        new_x[..., :-1] = x
        return new_x
def homogenize_ncoord(x, ncoord=2):
    assert (ncoord >= 1) and (int(ncoord) == ncoord), f"[homogenize_ndim]Error: required dimension is invalid: {ncoord}!"
    nc = int(ncoord)
    if x.shape[-1] == nc: # ordinary coordinates
        return homogenize(x)
    elif x.shape[-1] == (nc+1) and (x[:, -1] == 1).all(): # homogeneous coordinates
        return x
    else:
        print(f"[homogenize_ncoord]Error: Input has invalid shape {x.shape} or value!")
        exit(0)

def display_line_diff(infile, fig=None, ax=None):
    data = np.loadtxt(infile) # ndarray (N, 3)
    cbs = list(combinations(data, 2))
    angles = list()
    for cb in cbs:
        angles.append(acos(line_cos_angle(cb[0], cb[1]))*180/pi)
    angles = np.array(angles)
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    if fig is None:
        fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    ax.scatter(np.array([i+1 for i in range(len(cbs))]), angles, s=5, marker='+')
    ax.axhline(mean_angle)
    ax.set_title(f"{infile}\nmean = {mean_angle}\nstd = {std_angle}", fontsize=10)
    return fig, ax

def display_pts(infile, fig, ax):
    data = np.loadtxt(infile) # ndarray (N, 2)
    mean_pt = np.mean(data, axis=0)
    std_pt = np.std(data, axis=0)
    if fig is None:
        fig = None
    if ax is None:
        fig, ax = plt.subplots(2, 1)
    ax[0].scatter(np.array([i+1 for i in range(data.shape[0])]), data[:, 0], s=5, marker='+', c='y')
    ax[0].axhline(mean_pt[0], color='y')
    ax[0].set_title(f"{infile}\nx mean = {mean_pt[0]}\nx std = {std_pt[0]}", fontsize=8)
    ax[1].scatter(np.array([i+1 for i in range(data.shape[0])]), data[:, 1], s=5, marker='*', c='g')
    ax[1].axhline(mean_pt[1], color='g')
    ax[1].set_title(f"y mean = {mean_pt[1]}\ny std = {std_pt[1]}", fontsize=8)
    return fig, ax

def collect_pts(in_root, out_root):
    assert os.path.isdir(in_root), f"in_root {in_root} is not a valid directory!"
    assert os.path.isdir(out_root), f"out_root {out_root} is not a valid directory!"
    dirs = glob.glob(os.path.join(in_root, '*image.txt')) # N
    collector = dict()
    for idx, d in enumerate(dirs):
        data = np.loadtxt(d) # (8, 2)
        for r in range(data.shape[0]): # 8
            if not r in collector:
                collector[r] = np.zeros((len(dirs), data.shape[1])) # (N, 2)
            collector[r][idx, :] = data[r, :]
    for pt_idx in collector.keys():
        np.savetxt(os.path.join(out_root, str(pt_idx)+'.txt'), collector[pt_idx], fmt='%1.3f')
        print(f"{pt_idx}.txt saved!")

def compare_pts(root1, root2, fig=None, axes=None):
    assert os.path.isdir(root1), f"root1 {root1} is not a valid directory!"
    assert os.path.isdir(root2), f"root2 {root2} is not a valid directory!"
    dirs = glob.glob(os.path.join(root1, '*image.txt')) # N
    pts_dist = list()
    for idx, dir1 in enumerate(dirs):
        file_name = dir1.split('/')[-1]
        dir2 = os.path.join(root2, file_name)
        if not os.path.isfile(dir2): continue
        pts1 = np.loadtxt(dir1) # (8, 2)
        pts2 = np.loadtxt(dir2) # (8, 2)
        if pts1.shape != pts2.shape: continue
        pts_dist.append(dist(pts1, pts2)) # (8, )
    pts_dist = np.vstack(pts_dist) # (N', 8)
    ## display
    if fig is None:
        fig = None
    if axes is None:
        fig, axes = plt.subplots(2, 1)
    xvalues_img, xvalues_pts = np.mgrid[1:pts_dist.shape[0]+1:1, 1:pts_dist.shape[1]+1:1]
    axes[0].scatter(xvalues_img.reshape((-1,)), pts_dist.reshape((-1,)), c='y', zorder=5)
    axes[0].scatter(xvalues_img[:, 0], np.mean(pts_dist, axis=1), c='g', zorder=10)
    axes[0].axhline(np.mean(np.mean(pts_dist)), c='g')
    axes[0].set_xlabel('image')
    axes[0].set_ylabel('distance')
    axes[1].scatter(xvalues_pts.reshape((-1,)), pts_dist.reshape((-1,)), c='y', zorder=5)
    axes[1].scatter(xvalues_pts[0, :], np.mean(pts_dist, axis=0), c='g', zorder=10)
    axes[1].axhline(np.mean(np.mean(pts_dist)), c='g')
    axes[1].set_xlabel('point')
    axes[1].set_ylabel('distance')
    return fig, axes

if __name__ == '__main__':
    # fig, axes = plt.subplots(1, 2)
    # display_line_diff('before.txt', fig, axes[0])
    # display_line_diff('after.txt', fig, axes[1])
    # plt.show()
    # fig, axes = plt.subplots(2, 2)
    # display_pts('without.txt', fig, axes[:, 0])
    # display_pts('with.txt', fig, axes[:, 1])
    # plt.show()
    compare_pts('../../../seed4_before', '../../../seed4_after')
    plt.show()
