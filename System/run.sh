# !/usr/bin/env bash
# shellcheck disable=SC2154
# shellcheck disable=SC1091

## configure
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./src/utils/yaml.sh

tic=$(date +%s)
## load yaml parameters
if [ "$1" == "-p" ]; then
    PARAM_FILE=$2
else
    PARAM_FILE=param.yaml
fi
printf "Parameter Specification [${PARAM_FILE}]\n"
create_variables $PARAM_FILE

## get data paths
video_root=${DATA_video_root}
dlt_gt_dir=${DATA_dlt_gt_dir}
opt_gt_dir=${DATA_opt_gt_dir}
# echo $video_root
# echo $dlt_gt_dir
# echo $opt_gt_dir

## get roots
img_root=${ROOTS_save_root}${DEFAULTS_img_root}
if  [ ! -z ${ROOTS_img_root+x} ]; then img_root=${ROOTS_img_root}; fi
box_root=${ROOTS_save_root}${DEFAULTS_box_root}
if  [ ! -z ${ROOTS_box_root+x} ]; then box_root=${ROOTS_box_root}; fi
box_summary_dir=${ROOTS_save_root}${DEFAULTS_box_summary_dir}
if  [ ! -z ${ROOTS_box_summary_dir+x} ]; then box_summary_dir=${ROOTS_box_summary_dir}; fi
edge_root=${ROOTS_save_root}${DEFAULTS_edge_root}
if  [ ! -z ${ROOTS_edge_root+x} ]; then edge_root=${ROOTS_edge_root}; fi
edge_summary_dir=${ROOTS_save_root}${DEFAULTS_edge_summary_dir}
if  [ ! -z ${ROOTS_edge_summary_dir+x} ]; then edge_summary_dir=${ROOTS_edge_summary_dir}; fi
corner_root=${ROOTS_save_root}${DEFAULTS_corner_root}
if  [ ! -z ${ROOTS_corner_root+x} ]; then corner_root=${ROOTS_corner_root}; fi
corner_summary_dir=${ROOTS_save_root}${DEFAULTS_corner_summary_dir}
if  [ ! -z ${ROOTS_corner_summary_dir+x} ]; then corner_summary_dir=${ROOTS_corner_summary_dir}; fi
calibrate_root=${ROOTS_save_root}${DEFAULTS_calibrate_root}
if  [ ! -z ${ROOTS_calibrate_root+x} ]; then calibrate_root=${ROOTS_calibrate_root}; fi
filter_root=${ROOTS_save_root}${DEFAULTS_filter_root}
if  [ ! -z ${ROOTS_filter_root+x} ]; then filter_root=${ROOTS_filter_root}; fi

## create save folder
if [ ! -d "${ROOTS_save_root}" ]; then
    mkdir "${ROOTS_save_root}"
    printf "Output folder '${ROOTS_save_root}' created!\n"
else
    read -p "Output folder '${ROOTS_save_root}' exists! Continue to overwrite? [Y/n] " yn
    case $yn in 
        [Yy]* ) echo ;;
        * ) exit;;
    esac
    # printf "Output folder '${ROOTS_save_root}' exists! Overwrite!\n"
fi

## create log file
if $META_save_log; then
    LOG_FILE=${ROOTS_save_root}/auto-calib.log
    if [ -f "${LOG_FILE}" ]; then
        rm $LOG_FILE
    fi
    touch $LOG_FILE
    date>$LOG_FILE
    exec &> >(tee -a "${LOG_FILE}") # print stdout & stderror to the terminal, and store them to the log as well
fi
# exec 3>&1 1>>$LOG_FILE 2>&1
# exec 1>$LOG_FILE # stdout
# exec 2>&1        # stderr

## copy yaml parameters
if [ -f "${ROOTS_save_root}/${PARAM_FILE}" ]; then
    rm ${ROOTS_save_root}/${PARAM_FILE}
fi
cp ./${PARAM_FILE} ${ROOTS_save_root}/${PARAM_FILE}
printf "Parameter specification copied!\n"

## extract frames from videos
if $FRAME_EXTRACTOR_is_on; then
    printf "\n=== FRAME EXTRATOR start ===\n"
    if [ -d "${img_root}" ]; then
        rm -r ${img_root}/
    fi
    mkdir ${img_root}/
    extractor_tic=$(date +%s)
    python3 ./src/frame_extractor/extract_image.py $video_root $img_root $FRAME_EXTRACTOR_period $META_img_ext
    extractor_toc=$(date +%s)
    printf "\n=== FRAME EXTRATOR end $(($extractor_toc-$extractor_tic))s ===\n"
fi

## detect stop signs
if $DETECTOR_is_on; then
    printf "\n=== DETECTOR start ===\n"
    if [ -d "${box_root}" ]; then
        rm -r ${box_root}/
    fi
    mkdir ${box_root}/
    detector_tic=$(date +%s)
    python3 ./src/detector/detector.py $img_root $box_root $box_summary_dir $DETECTOR_model $DETECTOR_box_scale
    detector_toc=$(date +%s)
    printf "\n=== DETECTOR end $(($detector_toc-$detector_tic))s ===\n"
fi

## detect sub-pixel edge points
if $EDGE_is_on; then
    printf "\n=== SUB-PIXEL EDGE start ===\n"
    if [ -d "${edge_root}" ]; then
        rm -r ${edge_root}/
    fi
    mkdir ${edge_root}/
    edge_tic=$(date +%s)
    ./src/sub_pixel_edge/runSubPixelEdge $box_summary_dir ${box_root}/ $edge_summary_dir ${edge_root}/ $META_img_ext
    edge_toc=$(date +%s)
    printf "\n=== SUB-PIXEL EDGE end $(($edge_toc-$edge_tic))s ===\n"
fi

## estimate lines & find corners
if $LINE_ESTIMATE_is_on; then
    printf "\n=== LINE ESTIMATE start ===\n"
    if [ -d "${corner_root}" ]; then
        rm -r ${corner_root}/
    fi
    mkdir ${corner_root}/
    corner_args=(${META_img_ext} ${LINE_ESTIMATE_args_correctness_thresh} ${LINE_ESTIMATE_args_is_improved} ${LINE_ESTIMATE_args_pt_improve_thresh} ${LINE_ESTIMATE_args_line_improve_thresh} ${LINE_ESTIMATE_args_shape_check} ${LINE_ESTIMATE_args_is_brute_force} ${LINE_ESTIMATE_args_is_paral} ${LINE_ESTIMATE_args_debug})
    # python3 ./src/line_estimation/run.py $img_root $edge_root $edge_summary_dir $corner_summary_dir $corner_root ${LINE_ESTIMATE_args_is_improved} ${LINE_ESTIMATE_args_pt_improve_thresh} ${LINE_ESTIMATE_args_line_improve_thresh} ${LINE_ESTIMATE_args_is_brute_force} ${LINE_ESTIMATE_args_debug}
    corner_tic=$(date +%s)
    python3 ./src/line_estimation/run.py $img_root $edge_root $edge_summary_dir $corner_summary_dir $corner_root ${corner_args[@]}
    corner_toc=$(date +%s)
    printf "\n=== LINE ESTIMATE end $(($corner_toc-$corner_tic))s ===\n"
fi

## shuffle the order of corner files
calib_names_dir=$corner_summary_dir
if $CALIBRATE_is_on || $FILTER_is_on; then
    if $META_names_shuffled; then
        calib_names_dir=$corner_root/intersections_shuffled.txt
        python3 ./src/utils/shuffle_names.py $corner_summary_dir $calib_names_dir
    fi
fi

## calibrate
if $CALIBRATE_is_on; then
    printf "\n=== CALIBRATION start ===\n"
    if [ -d "${calibrate_root}" ]; then
        rm -r $calibrate_root
    fi
    mkdir $calibrate_root
    # mkdir ${calibrate_root}/Traffic_sign_auto
    calib_tic=$(date +%s)
    ./src/calibration/runCalibrate $calib_names_dir $dlt_gt_dir $opt_gt_dir ${img_root}/ ${corner_root}/ ${corner_root}/ ${calibrate_root}/ $META_img_ext
    calib_toc=$(date +%s)
    python3 ./src/calibration/vis_calib.py ${calibrate_root}/ $CALIBRATE_args_vis_on
    printf "\n=== CALIBRATION end $(($calib_toc-$calib_tic))s ===\n"
fi

## filter
if $FILTER_is_on; then
    printf "\n=== FILTER start ===\n"
    if [ -d "${filter_root}" ]; then
        rm -r $filter_root
    fi
    mkdir $filter_root
    # mkdir ${filter_root}/Traffic_sign_auto
    filter_args=(${FILTER_args_mode} ${FILTER_args_window_size} ${FILTER_args_system_noise} ${FILTER_args_measure_noise} ${FILTER_args_initial_cov})
    filter_tic=$(date +%s)
    # ./src/filter/runFilter $FILTER_args_mode $FILTER_args_window_size $FILTER_args_system_noise $FILTER_args_measure_noise $FILTER_args_initial_cov $corner_summary_dir $dlt_gt_dir ${img_root}/ ${corner_root}/ ${corner_root}/ ${filter_root}/
    ./src/filter/runFilter ${filter_args[@]} $calib_names_dir $dlt_gt_dir ${img_root}/ ${corner_root}/ ${corner_root}/ ${filter_root}/ $META_img_ext
    filter_toc=$(date +%s)
    python3 ./src/filter/vis_filter.py ${filter_root}/ $FILTER_args_vis_on $FILTER_args_mode
    printf "\n=== FILTER end $(($filter_toc-$filter_tic))s ===\n"
fi
toc=$(date +%s)
# TIME SUMMARY
echo ""
if $FRAME_EXTRACTOR_is_on; then
    printf "FRAME EXTRACTOR: $(($extractor_toc-$extractor_tic))s\n"
fi
if $DETECTOR_is_on; then
    printf "DETECTOR       : $(($detector_toc-$detector_tic))s\n"
fi
if $EDGE_is_on; then
    printf "SUB-PIXEL EDGE : $(($edge_toc-$edge_tic))s\n"
fi
if $LINE_ESTIMATE_is_on; then
    printf "LINE ESTIMATE  : $(($corner_toc-$corner_tic))s\n"
fi
if $CALIBRATE_is_on; then
    printf "CALIBRATION    : $(($calib_toc-$calib_tic))s\n"
fi
if $FILTER_is_on; then
    printf "FILTER         : $(($filter_toc-$filter_tic))s\n"
fi
echo "---"
printf "TOTAL TIME     : $(($toc-$tic))s\n"