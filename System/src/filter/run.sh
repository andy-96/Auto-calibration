#!/bin/sh

#argv1 -> mode     1:mode1;2:mode2;3:mode3
#mode1 -> N-1 images out of N sets  mode2 -> moving window mode3 -> incremental 

#argv2 -> window_size(it is used only if mode == 2)
#argv3 -> process_noise
#argv4 -> measurement_noise
#argv5 -> init_covariance
./Filter 3 6 1 1000 100 ../intersection_summary.txt ../Results/cell_phone/Traffic_sign_gt/intrinsics_dlt.xml ../raw_imgs/cell_phone/ ../ ../ ../Results/cell_phone/ 
python3 Draw_only_DLT.py ../Results/cell_phone/ 
