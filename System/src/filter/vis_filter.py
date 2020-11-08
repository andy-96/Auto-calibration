import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import sys
import os
save_dir=sys.argv[1]
vis_on = eval(sys.argv[2])
filter_mode = sys.argv[3]
try:
    kalman_fx = open(save_dir+"fx_kalman.txt","r")
    kalman_fx_error = open(save_dir+"fx_kalman_error.txt","r")
    kalman_fy = open(save_dir+"fy_kalman.txt","r")
    kalman_fy_error = open(save_dir+"fy_kalman_error.txt","r")
    fx_dlt, fy_dlt, error_fx_dlt, error_fy_dlt = [], [], [], []

    mea_fx = open(save_dir+"fx_measurement.txt","r")
    mea_fx_error = open(save_dir+"fx_measurement_error.txt","r")
    mea_fy = open(save_dir+"fy_measurement.txt","r")
    mea_fy_error = open(save_dir+"fy_measurement_error.txt","r")
    fx_mea, fy_mea, error_fx_mea, error_fy_mea = [], [], [], []

    f_groundtruth = open(save_dir+"kalman_groundtruth.txt","r")
except:
    print(f"[vis_filter.py]Error: Unable to find required texts under '{save_dir}':\n{sorted(os.listdir(save_dir))}")
    exit(0)

groundtruth = []
for line in f_groundtruth:
    groundtruth.append(float(line))
fx_gt_dlt = groundtruth[0]
fy_gt_dlt = groundtruth[1]

# for line in kalman_fx:
#     fx_dlt.append(float(line))
# fx_dlt_xtick = list(range(2, 2+len(fx_dlt)))
# for line in kalman_fy:
#     fy_dlt.append(float(line))
# fy_dlt_xtick = list(range(2, 2+len(fy_dlt)))
for line in kalman_fx_error:
    error_fx_dlt.append(float(line))
fx_dlt_xtick = list(range(2, 2+len(error_fx_dlt)))
#     error_fx_gt.append(0)
for line in kalman_fy_error:
    error_fy_dlt.append(float(line))
fy_dlt_xtick = list(range(2, 2+len(error_fy_dlt)))
#     error_fy_gt.append(0)

# for line in mea_fx:
#     fx_mea.append(float(line))
# fx_mea_xtick = list(range(2, 2+len(fx_mea)))
# for line in mea_fy:
#     fy_mea.append(float(line))
# fy_mea_xtick = list(range(2, 2+len(fy_mea)))
for line in mea_fx_error:
    error_fx_mea.append(float(line))
fx_mea_xtick = list(range(2, 2+len(error_fx_mea)))
for line in mea_fy_error:
    error_fy_mea.append(float(line))
fy_mea_xtick = list(range(2, 2+len(error_fy_mea)))

# mode
if int(filter_mode) == 1: mode = 'Smoothing'
elif int(filter_mode) == 2: mode = 'Sliding Window'
elif int(filter_mode) == 3: mode = 'Accumulation'
else: mode = 'Unknown'

plt.rcParams['figure.figsize']=(12.0,4.0)
fig, axes = plt.subplots(1, 2)
fig.subplots_adjust(wspace=0.4, left=0.065, right=0.93, bottom=0.15, top=0.95)
# fig.suptitle(f'Calibration Results with Kalman Filter\n{mode} Mode')

## fx ##
axes[0].plot(fx_dlt_xtick, error_fx_dlt, linewidth=2, color='r', zorder=10)
axes[0].plot(fx_mea_xtick, error_fx_mea, linewidth=2, color='b', zorder=5)
axes[0].axhline(y=0, color='k', linewidth=2)
axes[0].legend(labels = ['Filter', 'No Filter', f'GT={round(fx_gt_dlt)}'], loc = 2)
axes[0].set_ylabel("Relative Error of fx", fontsize=12)
axes[0].set_xlabel("Number of Images", fontsize=12)
axes[0].grid(which='both')
axins0 = axes[0].inset_axes([0.5, 0.35, 0.3, 0.5])
axins0.plot(fx_dlt_xtick, error_fx_dlt, linewidth=2, color='r', zorder=10)
axins0.plot(fx_mea_xtick, error_fx_mea, linewidth=2, color='b', zorder=5)
axins0.axhline(y=0, color='k', linewidth=2)
axins0.set_xlim(fx_dlt_xtick[-1]-35, fx_dlt_xtick[-1]+5)
axins0.set_ylim(error_fx_dlt[-1]-0.05, error_fx_dlt[-1]+0.05)
axes[0].indicate_inset_zoom(axins0)
# transformation functions between error & value
def fx2efx(x):
    return (x - fx_gt_dlt) / fx_gt_dlt
def efx2fx(x):
    return fx_gt_dlt * x + fx_gt_dlt
secax_fx = axes[0].secondary_yaxis('right', functions=(efx2fx, fx2efx))
secax_fx.set_ylabel("fx (px)", fontsize=12)

## fy ##
axes[1].plot(fy_dlt_xtick, error_fy_dlt, linewidth=2, color='r', zorder=10)
axes[1].plot(fy_mea_xtick, error_fy_mea, linewidth=2, color='b', zorder=5)
axes[1].axhline(y=0, color='k', linewidth=2)
axes[1].legend(labels = ['Filter', 'No Filter', f'GT={round(fy_gt_dlt)}'], loc = 2)
axes[1].set_ylabel("Relative Error of fy", fontsize=12)
axes[1].set_xlabel("Number of Images", fontsize=12)
axes[1].grid(which='both')
axins1 = axes[1].inset_axes([0.5, 0.35, 0.3, 0.5])
axins1.plot(fy_dlt_xtick, error_fy_dlt, linewidth=2, color='r', zorder=10)
axins1.plot(fy_mea_xtick, error_fy_mea, linewidth=2, color='b', zorder=5)
axins1.axhline(y=0, color='k', linewidth=2)
axins1.set_xlim(fy_dlt_xtick[-1]-35, fy_dlt_xtick[-1]+5)
axins1.set_ylim(error_fy_dlt[-1]-0.05, error_fy_dlt[-1]+0.05)
axes[1].indicate_inset_zoom(axins1)
# transformation functions between error & value
def fy2efy(x):
    return (x - fy_gt_dlt) / fy_gt_dlt
def efy2fy(x):
    return fy_gt_dlt * x + fy_gt_dlt
secax_fy = axes[1].secondary_yaxis('right', functions=(efy2fy, fy2efy))
secax_fy.set_ylabel("fy (px)", fontsize=12)

plt.savefig(save_dir+'Calibration_kalman.png')
if vis_on:
    plt.show()
