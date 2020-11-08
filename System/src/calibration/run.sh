#/bin/sh

./Calibration ../intersection_summary.txt ../Results/cell_phone/Traffic_sign_gt/intrinsics_dlt.xml  ../Results/cell_phone/Traffic_sign_gt/intrinsics_opt.xml ../raw_imgs/cell_phone/  ../ ../ ../Results/cell_phone/
python3 Draw_only_DLT.py ../Results/cell_phone/
