import sys, cv2, pickle
import numpy as np
import matplotlib.pyplot as plt
from os import path, mkdir
from tqdm import tqdm
from math import pi, cos, acos

from svd_ransac import generate_lines, get_components
from svd_paral import generate_lines_paral
from svd_estimate import display_lines, get_round_intersections

from improve import grad_improve
from utils import line_cos_angle
from generate_physical import get_physical_pts
from shape_matching import get_normalized_affine_arc, get_TI_Hausdorff_diff_1d

def get_segment_intersections(lines, terminals, thresh=2):
    assert len(lines) == len(terminals), f"[get_segment_intersections]: Lines and terminal points are not matched: {len(lines)} lines v.s. {len(terminals)} terminal pairs!"
    intersections = [] # [ndarray(2, ), ...]
    num = len(lines)
    for i in range(num):
        n1 = lines[i]
        ends_i = terminals[i]
        for j in range(i+1, num):
            n2 = lines[j]
            ends_j = terminals[j]
            A = np.array([
                [n1[0], n1[1]],
                [n2[0], n2[1]]
            ])
            v = np.array([
                [n1[2]],
                [n2[2]]
            ])
            try:
                intersection = (-np.linalg.inv(A) @ v).flatten()
            except: # parallel: n1 // n2
                continue
            if (np.sqrt(np.sum((ends_i-intersection)**2, axis=1)) < thresh).any() or\
                (np.sqrt(np.sum((ends_j-intersection)**2, axis=1)) < thresh).any(): # the intersection is close to one of the terminal points
                intersections.append(intersection)
    return np.array(intersections)

def get_nearest_intersections(lines, ref_intersections, thresh=10):
    assert ref_intersections.ndim == 2 and ref_intersections.shape[1] == 2, f"[get_nearest_intersections]: input 'ref_intersections' has wrong shape {ref_intersections.shape}!"
    num = ref_intersections.shape[0]
    intersections = [None for ii in range(num)] # [ndarray(2, ), ...]
    min_dist = [float('inf') for ii in range(num)]
    num_lines = len(lines)
    for i in range(num_lines):
        n1 = lines[i]
        ends = terminals[i]
        for j in range(i+1, num_lines):
            n2 = lines[j]
            A = np.array([
                [n1[0], n1[1]],
                [n2[0], n2[1]]
            ])
            v = np.array([
                [n1[2]],
                [n2[2]]
            ])
            intersection = (-np.linalg.inv(A) @ v).flatten()
            # assign the intersection to the closest reference intersection
            dist = np.sqrt(np.sum((ref_intersections-intersection)**2, axis=1))
            idx = np.argmin(dist)
            if min_dist[idx] > dist[idx] and dist[idx] <= thresh:
                min_dist[idx] = dist[idx]
                intersections[idx] = intersection
    intersections = list(filter((None).__ne__, intersections))
    if len(intersections) != num:
        print(f"[get_nearest_intersections]: Unmatched reference intersection exists!")
    return np.array(intersections)

def clockwise_sort(pts):
    assert (pts.ndim == 2) and (pts.shape[1] == 2), f"[clockwise_sort]: Invalid shape of input 'pts' {pts.shape}!"  
    center = np.mean(pts, axis=0)
    vec = pts - center
    rad = np.arctan2(vec[:, 1], vec[:, 0])
    sorted_idx = np.argsort(rad) # radius in ascending order
    return pts[sorted_idx, :]

# given the boundary points of any shape, (coarsely) find all integer points enclosed
def get_enclosed_points(boundary_pts):
    pts = boundary_pts.copy()
    assert (pts.ndim == 2) and (pts.shape[1] == 2), f"[get_enclosed_points]: Invalid shape of input 'pts' {pts.shape}!"  
    x = pts[:, 0]
    y = pts[:, 1]
    diameter = ((np.max(x)-np.min(x))**2 + (np.max(y)-np.min(y))**2)**(0.5)
    x += diameter
    y += diameter
    img_shape = (int(np.max(x)+diameter), int(np.max(y)+diameter))
    img = np.zeros(img_shape, dtype=np.uint8)
    img[x.astype(np.int), y.astype(np.int)] = 1
    kernel_size = int(diameter / 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    x, y = np.where(img == 1)
    return np.vstack([x, y]).T - diameter

def check_parallel(lines, angle_thresh=10, debug=False, img=None):
    unchecked = lines.copy()
    pairs = list()
    while len(unchecked) > 0:
        line1 = unchecked[0]
        max_cos = 0
        max_idx = -1
        for i in range(1, len(unchecked)): # find a line paired to line1
            line2 = unchecked[i]
            cos_theta = line_cos_angle(line1, line2)
            if cos_theta > max_cos:
                max_cos = cos_theta
                max_idx = i
        if max_idx == -1: return True, pairs
        if max_cos < abs(cos(angle_thresh*pi/180)): # unparallel pair
            if debug and img is not None:
                fig, ax = display_lines(img, [line1, unchecked[max_idx]])
                ax.imshow(img)
                ax.set_title(f"unparallel: {acos(max_cos)*180/pi}")
                plt.show()
            return False, None
        else: # parallel pair
            if debug and img is not None:
                fig, ax = display_lines(img, [line1, unchecked[max_idx]])
                ax.imshow(img)
                ax.set_title(f"parallel: {acos(max_cos)*180/pi}")
                plt.show()
            pairs.append([line1, unchecked[max_idx]])
            unchecked = unchecked[1:max_idx] + unchecked[(max_idx+1):]
    return True, pairs

def is_valid_shape(mode='parallel', **kargs):
    if mode.lower() == 'parallel':
        is_paired, pairs = check_parallel(kargs['lines'], angle_thresh=kargs['angle_thresh'], debug=kargs['debug'], img=kargs['img'])
        return is_paired
    elif mode.lower() == 'octagon':
        ## calculate the affine-invariant representation
        arc = get_normalized_affine_arc(kargs['corners'])
        # reference points
        pts_ref, _ = get_physical_pts(8, 0)
        pts_ref = pts_ref[:, :2]
        arc_ref = get_normalized_affine_arc(pts_ref)
        # calculate translation-invariant Hausdorff distances
        d, t = get_TI_Hausdorff_diff_1d(arc, arc_ref, 0.01)
        if d > kargs['Hausdorff_thresh']: return False
        else: return True
    
def main(img_root, file_root, in_summary_dir, out_summary_dir=None, save_root=None, img_tag='png', correctness_thresh=2, is_improved=True, pt_improve_thresh=-1, line_improve_thresh=-1, shape_check=1, is_brute_force=True, is_paral=False, debug=False):
    ### Hyper-parameters ###
    # img_tag = 'png'
    txt_names = ['inner_polygon'] # ['inner_T', 'inner_polygon', 'outer_polygon']

    # check roots
    if not path.isdir(img_root):
        print(f"{img_root} is not a valid directory!")
        exit(0)
    if not path.isdir(file_root):
        print(f"{file_root} is not a valid directory!")
    if (save_root is not None) and (not path.isdir(save_root)):
        mkdir(save_root)
    if out_summary_dir is not None:
        try:
            f = open(out_summary_dir, 'w')
            f.close()
        except:
            print(f"out_summary_dir '{out_summary_dir}' is invalid!")
            out_summary_dir = None

    with open(in_summary_dir, 'r') as f:
            names = f.readlines()

    for name in tqdm(names, desc=f'Stop Sign', position=1):
        name = name.strip()
        img_name = name.split('-')[0]
        # image preprocessing
        # img_name = name[:name.find('box')-1]
        # img = np.array(plt.imread(f'{img_root}/{img_name}.{img_tag}')) # RGB
        img = np.array(cv2.imread(f'{img_root}/{img_name}.{img_tag}'))[:, :, ::-1] # RGB

        fig = plt.figure()
        ax = fig.gca()
        total_num_pts = 0
        total_num_lines = 0
        total_num_intersections = 0
        ref_intersections = None
        # display range
        xmin, xmax = img.shape[1], 0
        ymin, ymax = img.shape[0], 0
        # bad results remove indicator
        is_abondoned = False
        # corners counter
        num_outer_polygon_corners = 0
        num_inner_polygon_corners = 0
        num_inner_T_corners = 0
        # corners for current image
        corners = list()
        for txt_name in txt_names:
            ### load points ###
            with open(f'{file_root}/{name}_{txt_name}.txt', 'r') as f:
                data = f.readlines()
            pts = []
            for d in data:
                d = d.strip()
                d_list = d.split(' ')
                pts.append((float(d_list[0]), float(d_list[1])))
            pts = np.array(pts, dtype=np.float)
            num_pts = pts.shape[0]
            total_num_pts += num_pts

            num_lines_required = 8
            if txt_name == 'inner_T':
                num_lines_required = 4
            # generate lines
            if is_brute_force: tqdm.write(f"[{name}]{txt_name}:BRUTE-FORCE")
            else: tqdm.write(f"[{name}]{txt_name}:RANSAC")
            if is_paral:
                lines, terminals = generate_lines_paral(pts, num_lines_required, dist_thresh=0.5, correctness_thresh=correctness_thresh, kernel_size=2, faster=not is_brute_force, keep_trace=False)
            else:
                lines, terminals = generate_lines(pts, num_lines_required, dist_thresh=0.5, correctness_thresh=correctness_thresh, kernel_size=2, faster=not is_brute_force, keep_trace=False)

            # display before improvement (for comparison)
            fig, ax = display_lines(img, lines, terminals, fig, ax, '--y')

            # get intersection points
            if txt_name == 'outer_polygon':
                num_outer_polygon_corners = 8
                intersections = clockwise_sort(get_nearest_intersections(lines, ref_intersections, thresh=10))
            if txt_name == 'inner_polygon':
                num_inner_polygon_corners = 8
                # find intersections
                intersections = get_segment_intersections(lines, terminals, thresh=10)
                intersections = clockwise_sort(intersections)
                # # check parallel
                # is_paired, pairs = check_parallel(lines, angle_thresh=7, debug=debug, img=img)
                # if not is_paired:
                #     tqdm.write(f"[{name}]inner_polygon:lines not paired!")
                #     is_abondoned = True
                #     break
                # shape check
                if shape_check == 1: # line parallel check
                    if not is_valid_shape(mode='parallel', lines=lines, angle_thresh=7, debug=debug, img=img):
                        tqdm.write(f"[{name}]inner_polygon:lines not paired!")
                        is_abondoned = True
                        break
                elif shape_check == 2: # octagon shape match
                    if not is_valid_shape(mode='octagon', corners=intersections, Hausdorff_thresh=0.02):
                        tqdm.write(f"[{name}]inner_polygon:octagon shape check fails!")
                        is_abondoned = True
                        break
                else:
                    raise NotImplementedError
                # check the number of intersections
                if intersections.ndim != 2 or intersections.shape[0] != 8:
                    tqdm.write(f"[{name}]inner_polygon:intersection shape {intersections.shape} wrong!")
                    is_abondoned = True
                    break
                ax.scatter(intersections[:, 0], intersections[:, 1], c='lightblue', s=5, zorder=10, marker='+')
                ## improve ##
                if is_improved: 
                    tqdm.write(f"[{name}]inner_polygon:IMPROVE {pt_improve_thresh if 0 <= pt_improve_thresh <= 1 else 'FREE'}|{line_improve_thresh if 0 <= line_improve_thresh <= 1 else 'FREE'}")
                    lines, terminals, num_improved_lines = grad_improve(img, lines, terminals, line_thresh=line_improve_thresh, pt_thresh=pt_improve_thresh)
                    tqdm.write(f"[{name}]inner_polygon:IMPROVED {num_improved_lines}/{len(lines)}")
                    # check parallelism
                    is_paired, pairs = check_parallel(lines, angle_thresh=7, debug=debug, img=img)
                    if not is_paired:
                        tqdm.write(f"[{name}]inner_polygon:lines not paired after improvement!")
                        is_abondoned = True
                        break
                    # find intersections/corners
                    intersections = get_segment_intersections(lines, terminals, thresh=10)
                    # check the number of intersections
                    if intersections.ndim != 2 or intersections.shape[0] != 8:
                        tqdm.write(f"[{name}]:intersection shape {intersections.shape} wrong after improvement!")
                        is_abondoned = True
                        break
                    intersections = clockwise_sort(intersections)
                ref_intersections = intersections

            if txt_name == 'inner_T':
                if len(lines) <= 2:
                    intersections = np.array([])
                    continue
                elif len(lines) == 3:
                    num_inner_T_corners = 2
                else:
                    num_inner_T_corners = 4
                tmp = get_round_intersections(lines)
                T_center = np.mean(pts, axis=0)
                dist = np.sqrt(np.sum((tmp - T_center)**2, axis=1))
                intersections = clockwise_sort(tmp[np.argpartition(dist, num_inner_T_corners)[:num_inner_T_corners], :])
                ax.scatter(intersections[:, 0], intersections[:, 1], c='lightblue', s=5, zorder=10, marker='+')

                ## improve ##
                if is_improved:
                    tqdm.write(f"[{name}]inner_T:IMPROVE")
                    lines, terminals, num_improved_lines = grad_improve(img, lines, terminals)
                    tqdm.write(f"[{name}]inner_T:IMPROVED {num_improved_lines}/{len(lines)}")
                tmp = get_round_intersections(lines)
                T_center = np.mean(pts, axis=0)
                dist = np.sqrt(np.sum((tmp - T_center)**2, axis=1))
                intersections = clockwise_sort(tmp[np.argpartition(dist, num_inner_T_corners)[:num_inner_T_corners], :])

            # display window range
            ixmin = np.min(intersections[:, 0])
            ixmax = np.max(intersections[:, 0])
            iymin = np.min(intersections[:, 1])
            iymax = np.max(intersections[:, 1])
            if ixmin < xmin: xmin = ixmin
            if ixmax > xmax: xmax = ixmax
            if iymin < ymin: ymin = iymin
            if iymax > ymax: ymax = iymax

            # count
            tqdm.write(f"[{name}]{txt_name}:({num_pts}, {len(lines)}, {len(intersections)})")
            total_num_lines += len(lines)
            total_num_intersections += len(intersections)

            # display
            fig, ax = display_lines(img, lines, terminals, fig, ax, '-g')
            # ax.scatter(pts[:, 0], pts[:, 1], c='g', s=1, zorder=1)
            ax.scatter(intersections[:, 0], intersections[:, 1], c='coral', s=5, zorder=10, marker='+')

            # store
            corners.append(intersections)
        ## DROP defective stop signs ##
        if is_abondoned:
            tqdm.write(f"[{name}]ABONDON")
            plt.close('all')
            tqdm.write("")
            continue

        # generate physical points
        physical_inner_polygon, physical_T = get_physical_pts(num_inner_polygon_corners, num_inner_T_corners)
        # image & object corner sets
        corners = np.vstack(corners)
        physicals = np.vstack([physical_inner_polygon, physical_T])
        assert corners.shape[0] == physicals.shape[0], f"[{name}]ERROR: Number of image & physical corners don't match: {corners.shape[0]} v.s. {physicals.shape[0]}!"
        
        ## SHAPE MATCHING ##
        # calculate the affine-invariant representation
        arc = get_normalized_affine_arc(corners)
        arc_ref = get_normalized_affine_arc(physicals[:, :2])
        # calculate translation-invariant Hausdorff distances
        d, t = get_TI_Hausdorff_diff_1d(arc, arc_ref, 0.01)
        if d > 0.02: 
            tqdm.write(f"[{name}]SHAPE UNMATCHED:{round(d, 5)}>0.02|{t}")
            tqdm.write(f"[{name}]ABONDON")
            plt.close('all')
            tqdm.write("")
            continue
        else: tqdm.write(f"[{name}]SHAPE MATCHED:{round(d, 5)}<=0.02|{t}")

        ## DISPLAY ##
        # crop image
        ax.imshow(img)
        ax.set_xlim(xmin-10, xmax+10)
        ax.set_ylim(ymin-10, ymax+10)
        ax.set_ylim(ax.get_ylim()[::-1])
        # set title
        ax.set_title(f'Points detected = {total_num_pts}\n Lines fit = {total_num_lines}\n Intersections = {total_num_intersections}', fontsize=10)
        
        ## SAVE text files & images ##
        if save_root is not None: 
            # save txt
            np.savetxt(f'{save_root}/{name}_image.txt', corners, fmt='%1.3f')
            np.savetxt(f'{save_root}/{name}_object.txt', physicals, fmt='%1.3f')
            # save image
            fig.savefig(f'{save_root}/{name}_estimate.{img_tag}')
        plt.close('all')
        
        ## WRITE summary file ##
        if out_summary_dir is not None:
            with open(f'{out_summary_dir}', 'a') as f:
                f.write(f'{name}\n')
        tqdm.write("")

if __name__ == '__main__':
    ### ROOTS & DIRECTORIES ###
    img_root = sys.argv[1]
    file_root = sys.argv[2]
    in_summary_dir = sys.argv[3]
    out_summary_dir = sys.argv[4]
    save_root = sys.argv[5]
    ### ARGUMENTS ###
    # shape_check: {1: line parallelism, 2: octagon shape matching}
    # kwargs = {
    #     'img_tag'               : 'png',
    #     'correctness_thresh'    : 25,
    #     'is_improved'           : True,
    #     'pt_improve_thresh'     : -1,
    #     'line_improve_thresh'   : 0.7,
    #     'shape_check'           : 1,
    #     'is_brute_force'        : True,
    #     'is_paral'              : True,
    #     'debug'                 : False
    # }
    img_ext = sys.argv[6]
    args = sys.argv[7:]
    args = [eval(arg) for arg in args]
    # np.random.seed(0)
    # main(img_root, file_root, in_summary_dir, out_summary_dir, save_root, **kwargs)
    main(img_root, file_root, in_summary_dir, out_summary_dir, save_root, img_ext, *args)
