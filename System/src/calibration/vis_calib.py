import matplotlib.pyplot as plt
import numpy as np
import sys
import os
save_dir = sys.argv[1]
vis_on = eval(sys.argv[2])
try:
    # f_fx_optim = open(save_dir+"fx_Optim.txt","r")
    # f_fy_optim = open(save_dir+"fy_Optim.txt","r")
    # f_error_fx_optim = open(save_dir+"error_fx_Optim.txt","r")
    # f_error_fy_optim = open(save_dir+"error_fy_Optim.txt","r")
    # fx_optim, fy_optim, error_fx_optim, error_fy_optim = [], [], [], []
    f_fx_dlt = open(save_dir+"fx_DLT.txt","r")
    f_fy_dlt = open(save_dir+"fy_DLT.txt","r")
    f_error_fx_dlt = open(save_dir+"error_fx_DLT.txt","r")
    f_error_fy_dlt = open(save_dir+"error_fy_DLT.txt","r")
    f_groundtruth = open(save_dir+"groundtruth.txt","r")
    fx_dlt, fy_dlt, error_fx_dlt, error_fy_dlt = [], [], [], []
except:
    print(f"[vis_calib.py]Error: Unable to find required texts under '{save_dir}':\n{sorted(os.listdir(save_dir))}")
    exit(0)

groundtruth = []
for line in f_groundtruth:
    groundtruth.append(float(line))
fx_gt_dlt = groundtruth[0]
fy_gt_dlt = groundtruth[1]
fx_gt_opt = groundtruth[2]
fy_gt_opt = groundtruth[3]
fx_groundtruth_dlt, fy_groundtruth_dlt, fx_groundtruth_opt, fy_groundtruth_opt, error_fx_gt, error_fy_gt = [], [], [], [], [], []
sum_x = 0
sum_y = 0
error_sum_x = 0
error_sum_y = 0
axis_origin_tmp = []
origin_start = 0
# axis_modified = []
for line in f_fx_dlt:
    fx_dlt.append(float(line))
    sum_x += float(line)
    fx_groundtruth_dlt.append(fx_gt_dlt)
    fx_groundtruth_opt.append(fx_gt_opt)
    axis_origin_tmp.append(origin_start)
    origin_start += 1
# axis_modified = [str(0)] * len(axis_origin)
# axis_modified[-1] = str(1)
# axis_modified_tmp = [str(x) for x in axis_origin_tmp]
axis_origin = []
axis_modified = []
num_exp = len(axis_origin_tmp)
for i in range(num_exp-1):
    if i % ((num_exp-2)//80+2) == 0:
        axis_origin.append(axis_origin_tmp[i])
        axis_modified.append(str(axis_origin_tmp[i]))
axis_origin.append(axis_origin_tmp[-1])
axis_modified.append('#')
# axis_modified[-1] = '*'
# for line in f_fy_optim:
#     fy_optim.append(float(line))
#     fy_groundtruth_dlt.append(fy_gt_dlt)
#     fy_groundtruth_opt.append(fy_gt_opt)
# for line in f_error_fx_optim:
#     error_fx_optim.append(float(line))
#     error_fx_gt.append(0)
# for line in f_error_fy_optim:
#     error_fy_optim.append(float(line))
#     error_fy_gt.append(0)
for line in f_fy_dlt:
    fy_dlt.append(float(line))
    sum_y += float(line)
for line in f_error_fx_dlt:
    error_fx_dlt.append(float(line))
    error_sum_x += float(line)
for line in f_error_fy_dlt:
    error_fy_dlt.append(float(line))
    error_sum_y += float(line)
sum_x /= len(fx_dlt)
sum_y /= len(fy_dlt)
error_sum_x /= len(error_fx_dlt)
error_sum_y /= len(error_fy_dlt)
x_tmp = [sum_x] * len(fx_dlt)
y_tmp = [sum_y] * len(fy_dlt)
error_x_tmp = [error_sum_x] * len(error_fx_dlt)
error_y_tmp = [error_sum_y] * len(error_fy_dlt)
plt.rcParams['figure.figsize']=(12.0,8.0)
plt.figure(1)
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.subplot(2,2,1)
#plt.plot(fx_optim,marker='*',linewidth=1,color='r')
plt.plot(fx_dlt,marker='+',linewidth=1,color='b')
plt.plot(fx_groundtruth_dlt,color='k',linewidth=1)
plt.plot(x_tmp, color = 'r', linewidth=1)
#plt.plot(fx_groundtruth_opt,color='r',linewidth=1)
#plt.legend(labels = ['optimalization', 'DLT', 'GT_DLT', 'GT_opt'], loc = 2)
plt.legend(labels = ['Result', 'GT','mean'], loc = 3)
plt.xlabel("Experiment")
plt.ylabel("fx")
plt.xticks(axis_origin,axis_modified, rotation='vertical')
#plt.title("fx:100-steps iteration for LM opt")
plt.grid()
# plt.suptitle('Auto-calibration results-Groundtruth obtained from chessboard calibration')
plt.suptitle(f'Calibration Results\n0~{len(axis_origin_tmp)-2}: One Stop Sign Removed\n#:All Stop Signs Used')
plt.subplot(2,2,2)
#plt.plot(fy_optim,marker='*', linewidth=1, color = 'r')
plt.plot(fy_dlt,marker='+',linewidth=1,color='b')
plt.plot(fy_groundtruth_dlt,color='k',linewidth=1)
plt.plot(y_tmp, color = 'r', linewidth=1)
#plt.plot(fy_groundtruth_opt,color='r',linewidth=1)
#plt.legend(labels = ['optimalization', 'DLT', 'GT_DLT', 'GT_opt'], loc = 2)
plt.legend(labels = ['Result', 'GT','mean'], loc = 3)
plt.xlabel("Experiment")
plt.ylabel("fy")
plt.xticks(axis_origin,axis_modified, rotation='vertical')
#plt.title("fy:100-steps iteration for LM opt")
plt.grid()
plt.subplot(2,2,3)
#plt.plot(error_fx_optim,marker='*',linewidth=1,color = 'r')
plt.plot(error_fx_dlt,marker='+',linewidth=1,color='b')
plt.plot(error_fx_gt,color='k',linewidth=1)
plt.plot(error_x_tmp, color = 'r', linewidth=1)
#plt.legend(labels = ['optimalization', 'DLT', 'Groundtruth'], loc = 2)
plt.legend(labels = ['Result', 'GT', 'mean'], loc = 3)
plt.xlabel("Experiment")
plt.ylabel("Error of fx")
plt.xticks(axis_origin,axis_modified, rotation='vertical')
#plt.title("Error of x:100-steps iteration for LM opt")
plt.grid()
plt.subplot(2,2,4)
#plt.plot(error_fy_optim,marker='*',linewidth=1,color='r')
plt.plot(error_fy_dlt,marker='+',linewidth=1,color='b')
plt.plot(error_fy_gt,color='k',linewidth=1)
plt.plot(error_y_tmp, color = 'r', linewidth=1)
#plt.legend(labels = ['optimalization', 'DLT', 'Groundtruth'], loc = 2)
plt.legend(labels = ['Result', 'GT','mean'], loc = 3)
plt.xlabel("Experiment")
plt.ylabel("Error of fy")
plt.xticks(axis_origin,axis_modified, rotation='vertical')
#plt.title("Error of y:100-steps iteration for LM opt")
plt.grid()
plt.figure(1).savefig(save_dir+'Calibration_results.png')
if vis_on:
    plt.show()
    plt.figure(2)
