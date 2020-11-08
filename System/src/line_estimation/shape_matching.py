import numpy as np
import matplotlib.pyplot as plt

def get_normalized_affine_arc(pts): # (N, 2)
    assert pts.ndim == 2 and pts.shape[1] == 2, f"[get_normalized_affine_arc]Error: input 'pts' has invalid shape {pts.shape}!"
    num_pts = pts.shape[0] # N
    affine_arcs = np.zeros((num_pts, )) # (N, )
    for i in range(num_pts): # each affine arc 'sigma'
        sigma = 0
        for k in range(i):
            k1 = (k + 1) % num_pts # k + 1
            k2 = (k + 2) % num_pts # k + 2
            p_k, p_k1, p_k2 = pts[[k, k1, k2], :] # p_k, p_{k+1}, p_{k+2}
            det_value = np.linalg.det(np.vstack([p_k1-p_k, p_k2-2*p_k1+p_k]).T)
            if det_value < 0:
                sigma += -abs(det_value)**(1./3)
            else:
                sigma += det_value**(1./3)
        affine_arcs[i] = sigma
    affine_arcs /= (affine_arcs[-1] + 1e-8)
    return affine_arcs

# shift all point in 'pts' by 'v' on the unit circle
def shift_on_unit_circle(pts, v): 
    if isinstance(v, (float, int)) or v.shape == (1, ):
        assert pts.ndim == 1 or pts.shape[-1] == 1, f"[shift_on_unit_circle]Error: shapes of inputs don't match!"
        return (pts + v) % (1+1e-8)
    else:
        v = v.flatten()
        assert pts.shape[-1] == v.shape[0], f"[shift_on_unit_circle]Error: shapes of inputs don't match!"
        return (pts + v) % (np.ones(v.shape)+1e-8)

r''' h(A, B)
    Inputs:
        A: ndarray (m, )
        B: ndarray (n, )
    Return:
        max_{a\in A} min_{b\in B} |a - b|
'''
def max_min_1d(A, B):
    assert A.ndim == 1 and B.ndim == 1, f"[max_min_1d]Error: input sets A and/or B have invalid shape(s): {A.shape} and/or {B.shape}!"
    rho = np.abs(A.reshape((-1, 1)) - B) # (m, n)
    return rho.min(axis=1).max(axis=0)

r''' H(A, B), Hausdorrf distance
    Inputs:
        A: ndarray (m, )
        B: ndarray (n, )
    Return:
        max{h(A, B), h(B, A)}
'''
def get_Hausdorff_dist_1d(A, B):
    return max(max_min_1d(A, B), max_min_1d(B, A))

r''' Discretized version of D(A, B), translation-invariant Hausdorff distance
    Inputs:
        A                     : ndarray (m, )
        B                     : ndarray (n, )
        translation_resolution: float \in (0, 1]
    Return:
        min_{t} H(A, B\bigoplus t)
        where t in range(0, 1, step=translation_resolution)
'''
def get_TI_Hausdorff_diff_1d(A, B, translation_resolution=0.01):
    assert 0 < translation_resolution <= 1, f"[get_TI_Hausdorff_diff_1d]Error: invalid translation resolution: {translation_resolution}!"
    min_dist = float('inf')
    best_translation = -1
    t = 0
    while t < 1:
        translated_dist = get_Hausdorff_dist_1d(A, shift_on_unit_circle(B, t))
        if translated_dist < min_dist:
            min_dist = translated_dist
            best_translation = t
        t += translation_resolution
    return min_dist, best_translation

if __name__ == '__main__':
    ### ROOTS ###
    exp_root = '../../exp2/3seq'
    corner_root = f'{exp_root}/corner_pts'
    summary_dir = f'{exp_root}/corner_pts/intersections.txt'
    save_root = f'{exp_root}/calibrate'

    ### read summary file
    with open(summary_dir, 'r') as f:
        names = f.readlines()
    ### translation-invariant Hausdorff distances
    distances = []
    translations = []
    for name in names:
        name = name.strip()
        ### load points
        with open(f'{corner_root}/{name}_image.txt', 'r') as f:
            data = f.readlines()
        pts = []
        for d in data:
            d = d.strip()
            d_list = d.split(' ')
            pts.append((float(d_list[0]), float(d_list[1])))
        pts = np.array(pts, dtype=np.float)
        ### load reference points
        with open(f'{corner_root}/{name}_object.txt', 'r') as f:
            data_ref = f.readlines()
        pts_ref = []
        for d_ref in data_ref:
            d_ref = d_ref.strip()
            d_ref_list = d_ref.split(' ')
            pts_ref.append((float(d_ref_list[0]), float(d_ref_list[1])))
        pts_ref = np.array(pts_ref, dtype=np.float)
        ### calculate the affine-invariant representation
        arc = get_normalized_affine_arc(pts)
        arc_ref = get_normalized_affine_arc(pts_ref)
        ### calculate translation-invariant Hausdorff distances
        d, t = get_TI_Hausdorff_diff_1d(arc, arc_ref, 0.01)
        print(f"[{name}] {round(d, 5)} | {t}")
        distances.append(d)
        translations.append(t)
    ### draw distances & translations
    assert len(distances) == len(translations), f"Not equal: number of distances v.s. number of translations"
    num_set = len(distances)
    ticks = []
    for i in range(num_set):
        if i % ((num_set-2)//80 + 2) == 0:
            ticks.append(i)
    fig, axes = plt.subplots(nrows=2, ncols=1)
    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.setp(axes, xticks=ticks, xticklabels=ticks)
    plt.setp(axes[0].get_xticklabels(), rotation='vertical')
    plt.setp(axes[1].get_xticklabels(), rotation='vertical')
    axes[0].grid()
    axes[1].grid()
    axes[0].plot(distances, marker='+')
    axes[0].set_ylabel("Hausdorff Distance")
    axes[0].set_xlabel("Set")
    axes[0].axhline(sum(distances)/len(distances), color='r', linewidth=0.5)
    axes[1].plot(translations, marker='+')
    axes[1].set_ylabel("Translation")
    axes[1].set_xlabel("Set")
    axes[1].axhline(sum(translations)/len(translations), color='r', linewidth=0.5)
    plt.show()
    fig.savefig(f'{save_root}/Hausdorff.png')
    plt.close('all')
