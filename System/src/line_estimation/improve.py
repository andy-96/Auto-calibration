import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from svd_estimate import display_lines
from scipy import interpolate
from svd_ransac import generate_lines
from svd_paral import generate_lines_paral
from utils import dist as get_dist

## Meta ##
W2L = 1.9 / 29 # width to length ratio of stop sign white borders

def get_dividing_pts_and_lines(line, terminal, num=8):
    # line    : [a, b, c] with line function ax + by + c = 0
    #           ndarray(3, )
    # terminal: [[xs, ys]; [xe, ye]]
    #           ndarray(2, 2)
    a, b, c = line
    (xs, ys), (xe, ye) = terminal[0, :], terminal[1, :]
    pts = []
    lines = []
    for k in range(1, num+1, 1): # {1, 2, ..., num}
        if abs(a) <= abs(b): # flatten slope
            xk = min(xs, xe) + k * abs(xs-xe) / (num+1)
            yk = -(a*xk + c) / b
        else: # steep slope
            yk = min(ys, ye) + k * abs(ys-ye) / (num+1)
            xk = -(b*yk + c) / a
        ak = -b
        bk = a
        ck = b * xk - a * yk
        pts.append(np.array([xk, yk]))
        lines.append(np.array([ak, bk, ck]))
    return pts, lines

def get_gradient_func(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, 0)
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.maximum(np.absolute(gradx), np.absolute(grady))
    grad_func = interpolate.interp2d(np.arange(img.shape[1]), np.arange(img.shape[0]), grad)
    return grad_func

# display the image gradient value of 'img' along the 'line' around (+/-'delta') the point 'pt'
def display_grad_1d(img, line, pt, delta=5, grad_func=None):
    # img : rgb, ndarray(?, ?, 3)
    # line: (a, b, c) where ax + by + c = 0, ndarray(3, )
    # pt  : (x, y), ndarray(2, )
    if grad_func is None:
        grad_func = get_gradient_func(img)
    fig, axes = plt.subplots(2, 1)
    x, y = pt
    a, b, c = line
    if abs(a) <= abs(b): # flatten slope
        xs = np.arange(x-delta, x+delta, 0.1)
        ys = - (a * xs + c) / b
    else: # steep slope
        ys = np.arange(y-delta, y+delta, 0.1)
        xs = - (b * ys + c) / a
    # distances to the center point
    ds = np.sqrt((xs - x)**2 + (ys - y)**2)
    ds[:len(ds)//2] *= -1
    # gradient
    gs = np.array([grad_func(xx, yy) for xx, yy in zip(xs, ys)]).flatten()
    # display
    fig, axes[0] = display_lines(img, [line], [np.array([[xs[0], ys[0]], [xs[-1], ys[-1]]])], fig=fig, ax=axes[0])
    axes[0].scatter(x, y, c='y', s=5, zorder=10, marker='+')
    axes[0].imshow(img)
    axes[0].set_xlim(min(xs[0], xs[-1])-10, max(xs[0], xs[-1])+10)
    axes[0].set_ylim(min(ys[0], ys[-1])-10, max(ys[0], ys[-1])+10)
    axes[0].set_ylim(axes[0].get_ylim()[::-1])
    axes[0].set_title('original image')
    if abs(a) <= abs(b): # flatten slope
        axes[1].plot(ds, gs, 'k-')
        axes[1].axvline(0, color='y')
        axes[1].set_xlabel('distances to yellow point (left to right)')
    else: # steep slope
        axes[1].plot(ds, gs, 'k-')
        axes[1].axvline(0, color='y')
        axes[1].set_xlabel('distances to yellow point (top to bottom)')
    axes[1].set_ylabel('image gradient magnitude')
    return fig, axes

# display the gradient ('grad_func') of image 'img' alone the line 'dl' centered at the point 'dpt' with the improved point 'pt'
def display_improved_grad(img, dpt, dl, pt, grad_func=None):
    fig, axes = display_grad_1d(img, dl, dpt, grad_func=grad_func)
    axes[0].scatter(pt[0], pt[1], c='g', s=5, zorder=10, marker='+')
    dist = np.sqrt(np.sum((pt - dpt)**2))
    if abs(dl[0]) <= abs(dl[1]): # flatten slope
        dist *= np.sign(pt[0] - dpt[0])
        axes[1].axvline(dist, color='g')
    else: # steep slope
        dist *= np.sign(pt[1] - dpt[1])
        axes[1].axvline(dist, color='g')
    return fig, axes

# get the point with (locally) maximum gradient along 'line' around the point 'pt'
def get_local_max_grad_1d(grad_func, line, pt, resolution=0.1):
    x, y = pt
    a, b, c = line
    max_grad1, max_grad2 = grad_func(x, y), grad_func(x, y)
    target1, target2 = np.array((x, y)), np.array((x, y))
    drop_cnt1, drop_cnt2 = 0, 0
    center = False
    idx = 1
    while (drop_cnt1 < int(1/resolution)) or (drop_cnt2 < int(1/resolution)):
        if abs(a) <= abs(b): # flatten slope
            x1 = x - idx * resolution
            x2 = x + idx * resolution
            y1 = - (a * x1 + c) / b
            y2 = - (a * x2 + c) / b
        else: # steep slope
            y1 = y - idx * resolution
            y2 = y + idx * resolution
            x1 = - (b * y1 + c) / a
            x2 = - (b * y2 + c) / a
        grad1, grad2 = grad_func(x1, y1), grad_func(x2, y2)
        if (idx == 1) and (grad_func(x, y) > max(grad1, grad2)): center = True
        if drop_cnt1 < int(1/resolution):
            if (grad1 > max_grad1):
                max_grad1 = grad1
                target1 = np.array((x1, y1))
                drop_cnt1 = 0
            else:
                drop_cnt1 += 1
        if drop_cnt2 < int(1/resolution):
            if (grad2 > max_grad2):
                max_grad2 = grad2
                target2 = np.array((x2, y2))
                drop_cnt2 = 0
            else:
                drop_cnt2 += 1
        idx += 1
    # dist1 = np.sqrt(np.sum((target1 - np.array((x, y)))**2))
    # dist2 = np.sqrt(np.sum((target2 - np.array((x, y)))**2))
    dist1 = get_dist(target1, np.array((x, y)))
    dist2 = get_dist(target2, np.array((x, y)))
    if dist1 == dist2:
        if max_grad1 > max_grad2: return target1
        else: return target2
    elif dist1 > dist2:
        if dist2 > 0 or center: return target2
        else: return target1
    else:
        if dist1 > 0 or center: return target1
        else: return target2

# display 'lines' centered at 'centers' of 'radius' half length
def display_centered_lines(img, lines, centers, radius=5, fig=None, ax=None):
    # lines  : [ndarray(3, ), ...]
    # centers: [ndarray(2, ), ...]

    # fig & ax
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    assert len(lines) == len(centers), f"[display_centered_lines]: the number of lines {len(lines)} doesn't match the number of centers {len(centers)}!"
    terminals = list()
    for line, pt in zip(lines, centers):
        x, y = pt
        a, b, c = line
        if abs(a) <= abs(b): # flatten slope
            x1, x2 = x-radius, x+radius
            y1, y2 = - (a * x1 + c) / b, - (a * x2 + c) / b
        else: # steep slope
            y1, y2 = y-radius, y+radius
            x1, x2 = - (b * y1 + c) / a, - (b * y2 + c) / a
        terminals.append(np.array([
            [x1, y1],
            [x2, y2]
        ]))
    # centers
    centers = np.vstack(centers)
    ax.scatter(centers[:, 0], centers[:, 1], c='y', s=5, zorder=10, marker='+')
    # lines
    fig, ax = display_lines(img, lines, terminals, fig, ax, '-k')
    return fig, ax
        

# given lines and terminals, fine-tune the line functions according to the image gradient magnitude
# pt_thresh, line_thresh: [0, 1] - (roughly) what percentage of the white stop sign border width can an improved point move from the original point's position
#                         other  - an improved point can move freely from the original point's position
#                         (white border: width / length = 1.9 / 29)
def grad_improve(img, lines, terminals, num_dpts=8, line_thresh=-1, pt_thresh=-1):
    assert len(lines) == len(terminals), f"[grad_improve]: # of lines ({len(lines)}) != # of terminal pairs ({len(terminals)})!"
    grad_func = get_gradient_func(img)
    new_lines, new_terminals = list(), list()
    num_improved_lines = 0
    for line, terminal in zip(lines, terminals):
        # line, terminal = lines[7], terminals[7]
        new_pts = list()
        dpts, dls = get_dividing_pts_and_lines(line, terminal, num=num_dpts) # dividing points & lines
        for dpt, dl in zip(dpts, dls): # improve sampled points on vertical direction
            new_pt = get_local_max_grad_1d(grad_func, dl, dpt) # improved point
            # fig, axes = display_improved_grad(img, dpt, dl, new_pt, grad_func)
            # plt.show()
            new_pts.append(new_pt)
            # exit()
        new_pts = np.vstack(new_pts) # improved points for the 'line'
        ## estimation of the white border width ##
        segment_length = get_dist(terminal[0, :], terminal[1, :])
        segment_width  = W2L * segment_length
        ## check point threshold: any new point should be bounded by the threshold, otherwise the original point is kept ##
        if (0 <= pt_thresh <= 1):
            outlider_idx = get_dist(dpts, new_pts) > (pt_thresh * segment_width)
            new_pts[outlider_idx, :] = np.vstack(dpts)[outlider_idx, :]
        ## check line threshold: number of improved points within threshold should be more than half ##
        if (0 <= line_thresh <= 1) and \
            (np.sum(get_dist(dpts, new_pts) <= (line_thresh * segment_width)) <= 0.5 * num_dpts): # less than half - abondon improvement
            new_line, new_terminal = line.copy(), terminal.copy()
        else: # keep improvement
            [new_line], [new_terminal] = generate_lines(new_pts, 1, dist_thresh=0.5, correctness_thresh=2, kernel_size=2, faster=False, keep_trace=False)
            num_improved_lines += 1
        ## terminal adjustment ##
        if abs(new_line[0]) <= abs(new_line[1]): # flatten slope
            xmin = min(terminal[0, 0], terminal[1, 0], new_terminal[0, 0], new_terminal[1, 0])
            xmax = max(terminal[0, 0], terminal[1, 0], new_terminal[0, 0], new_terminal[1, 0])
            new_terminal = np.array([
                [xmin, - (new_line[0]*xmin + new_line[2]) / new_line[1]],
                [xmax, - (new_line[0]*xmax + new_line[2]) / new_line[1]]
            ])
        else: # steep slope
            ymin = min(terminal[0, 1], terminal[1, 1], new_terminal[0, 1], new_terminal[1, 1])
            ymax = max(terminal[0, 1], terminal[1, 1], new_terminal[0, 1], new_terminal[1, 1])
            new_terminal = np.array([
                [- (new_line[1]*ymin + new_line[2]) / new_line[0], ymin],
                [- (new_line[1]*ymax + new_line[2]) / new_line[0], ymax]
            ])
        new_lines.append(new_line)
        new_terminals.append(new_terminal)
    return new_lines, new_terminals, num_improved_lines

if __name__ == '__main__':
    # load image
    img = np.array(plt.imread(f'improve/camera6_2_6.jpg')) # RGB
    # calculate gradient magnitude
    grad_func = get_gradient_func(img)
    
    # load lines & terminals
    with open('improve/camera6_2_6-0_inner_T.pkl', 'rb') as f:
        data = pickle.load(f)
    lines = data['lines']
    terminals = data['terminals']

    new_lines, new_terminals = list(), list()
    fig = plt.figure()
    ax = fig.gca()
    for line, terminal in zip(lines, terminals):
        line, terminal = lines[2], terminals[2]
        dpts, dls = get_dividing_pts_and_lines(line, terminal, num=8) # dividing points & lines
        new_pts = list()
        for dpt, dl in zip(dpts, dls):
            pt = get_local_max_grad_1d(grad_func, dl, dpt) # improved point
            display_improved_grad(img, dpt, dl, pt, grad_func)
            plt.show()
            continue
            new_pts.append(pt)
            ax.scatter(dpt[0], dpt[1], c='y', s=5, zorder=1, marker='+') # original point
            ax.scatter(pt[0], pt[1], c='g', s=5, zorder=1, marker='+')   # improved point
        exit()
        [new_line], [new_terminal] = generate_lines(np.vstack(new_pts), 1, 0.5, 0, 2, faster=True)
        # terminal adjustment
        if abs(new_line[0]) <= abs(new_line[1]): # flatten slope
            xmin = min(terminal[0, 0], terminal[1, 0], new_terminal[0, 0], new_terminal[1, 0])
            xmax = max(terminal[0, 0], terminal[1, 0], new_terminal[0, 0], new_terminal[1, 0])
            new_terminal = np.array([
                [xmin, - (new_line[0]*xmin + new_line[2]) / new_line[1]],
                [xmax, - (new_line[0]*xmax + new_line[2]) / new_line[1]]
            ])
        else: # steep slope
            ymin = min(terminal[0, 1], terminal[1, 1], new_terminal[0, 1], new_terminal[1, 1])
            ymax = max(terminal[0, 1], terminal[1, 1], new_terminal[0, 1], new_terminal[1, 1])
            new_terminal = np.array([
                [- (new_line[1]*ymin + new_line[2]) / new_line[0], ymin],
                [- (new_line[1]*ymax + new_line[2]) / new_line[0], ymax]
            ])
        new_lines.append(new_line)
        new_terminals.append(new_terminal)
    fig, ax = display_lines(img, new_lines, new_terminals, fig, ax)
    terms = np.vstack(terminals)
    ax.set_xlim(np.min(terms[:, 0])-10, np.max(terms[:, 0])+10)
    ax.set_ylim(np.min(terms[:, 1])-10, np.max(terms[:, 1])+10)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.imshow(img)
    plt.show()

