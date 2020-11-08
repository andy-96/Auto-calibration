#include<iostream>
#include<string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <random>
#include "Kalman_filter.h"
#include "Calibration.h"
using namespace std;
using namespace cv;
int main(int argc, char** argv) {
    //adjustable
    string tmp0 = string(argv[1]);
    string tmp = String(argv[2]);
    string tmp1 = String(argv[3]);
    string tmp2 = String(argv[4]);
    string tmp3 = String(argv[5]);
    //adjustable
    String intersection = String(argv[6]); //intersection_summary.txt
    String gt_dlt = String(argv[7]); //groundtruth for DLT method
    String raw_img_dir = String(argv[8]); //raw_image_dir
    String imageP_dir = String(argv[9]); //image_points_dir
    String objectP_dir = String(argv[10]); //object_points_dir
    String save_dir = String(argv[11]); //store directory
    String img_format = String(argv[12]); //img_format
    int scalar = 100; //scalar for filtering out bad measurements
    int total_number = 0;
    int window_size = stoi(tmp);
    int process_noise = stoi(tmp1);
    int measurement_noise = stoi(tmp2);
    int init_cova = stoi(tmp3);
    int mode = stoi(tmp0);
    //Read image name
    vector<String> file_list;
    fstream outFile;
    char buffer[256];
    outFile.open(intersection, ios::in);
    while(!outFile.eof())
    {
        outFile.getline(buffer,256,'\n');//getline(char *,int,char)
        if(String(buffer).size() >= 1) {
            file_list.push_back(String(buffer));
            total_number++;
        }
    }
    if(file_list.size() == 0){
       cout << "No images are found in the given directory!" << endl;
       return 0;
    }
    raw_img_dir += file_list[0].substr(0, file_list[0].length() - 2);
    raw_img_dir += String("."); //image points
    raw_img_dir += img_format; 
    Mat img = imread(raw_img_dir);
    Size img_size = img.size();
    calibration calib(file_list, window_size, img_size, total_number, mode);
    //calib.debug();
    cv::FileStorage fs1(gt_dlt, cv::FileStorage::READ);
    cv::Mat intrinsic_matrix_gt_dlt;
    fs1["camera_matrix"] >> intrinsic_matrix_gt_dlt;
    double fx_dlt_ = intrinsic_matrix_gt_dlt.at<double>(0, 0);
    double fy_dlt_ = intrinsic_matrix_gt_dlt.at<double>(1, 1);
    int tmp_;
    if(mode == 1){
        tmp_ = file_list.size() - 1;
        if(tmp_ < 2){
             cout << "Mode1:Not enough images for calibration!" << endl;
             cout << "The number of images is:" << file_list.size() << endl;
             return 0;
	}
    }
    else if(mode == 2){
        tmp_ = window_size;
	if(file_list.size() < tmp_){
     	    cout << "Mode2:Not enough images for calibration!" << endl;
	    cout << "The number of images is:" << file_list.size() << endl;
            cout << "The size of window is:" << window_size << endl;
            return 0;
	}
    }
    else{
        tmp_ = 2;
	if(file_list.size() < tmp_){
	    cout << "Mode3:Not enough images for calibration!" << endl;
            cout << "The number of images is:" << file_list.size() << endl;
	    return 0;
	}
    }
    kalman filter(mode, tmp_, scalar);
    filter.set_groundtruth(fx_dlt_, fy_dlt_);
    filter.noise_set(process_noise,measurement_noise); //1 -> Q;5 -> R
    if(mode == 1){
        for(int i = 0; i != file_list.size(); i++){
            if(i == 0){
                calib.update_points(imageP_dir, objectP_dir);
                calib.Calibration_camera();
                vector<double> fx_fy = calib.return_results();
                filter.init(fx_fy[0], fx_fy[1], init_cova); //initial covariance
                //this value would be smaller with iteration times being larger
                cout << "current measurement value:\n" << filter.current_measurement() << endl;
                cout << "current mean value:\n" << filter.current_mean() << endl;
                cout << "current covariance matrix:\n" << filter.current_covar() << endl;
            } else{
                calib.update_points(imageP_dir, objectP_dir);
                calib.Calibration_camera();
                vector<double> fx_fy = calib.return_results();
                filter.measurement(fx_fy[0], fx_fy[1]);
                int flag = filter.update();
                cout << "current measurement value:\n" << filter.current_measurement() << endl;
                cout << "current mean value:\n" << filter.current_mean() << endl;
                cout << "current covariance matrix:\n" << filter.current_covar() << endl;
            }
        }
    }
    else if(mode == 2) {
        while (!calib.quit()) {
            if (calib.check_first_set()) {
                calib.update_points(imageP_dir, objectP_dir);
                calib.Calibration_camera();
                vector<double> fx_fy = calib.return_results();
                filter.init(fx_fy[0], fx_fy[1], init_cova); //initial covariance
                //this value would be smaller with iteration times being larger
                cout << "current measurement value:\n" << filter.current_measurement() << endl;
                cout << "current mean value:\n" << filter.current_mean() << endl;
                cout << "current covariance matrix:\n" << filter.current_covar() << endl;
            } else {
                calib.update_points(imageP_dir, objectP_dir);
                calib.Calibration_camera();
                vector<double> fx_fy = calib.return_results();
                filter.measurement(fx_fy[0], fx_fy[1]);
                int flag = filter.update();
                cout << "current measurement value:\n" << filter.current_measurement() << endl;
                cout << "current mean value:\n" << filter.current_mean() << endl;
                cout << "current covariance matrix:\n" << filter.current_covar() << endl;
            }
        }
    }
    else if(mode == 3){
        for(int i = 0; i != file_list.size() - 1; i++){
            if(i == 0){
                calib.update_points(imageP_dir, objectP_dir);
                calib.Calibration_camera();
                vector<double> fx_fy = calib.return_results();
                filter.init(fx_fy[0], fx_fy[1], init_cova); //initial covariance
                //this value would be smaller with iteration times being larger
                cout << "current measurement value:\n" << filter.current_measurement() << endl;
                cout << "current mean value:\n" << filter.current_mean() << endl;
                cout << "current covariance matrix:\n" << filter.current_covar() << endl;
            } else{
                calib.update_points(imageP_dir, objectP_dir);
                calib.Calibration_camera();
                vector<double> fx_fy = calib.return_results();
                filter.measurement(fx_fy[0], fx_fy[1]);
                int flag = filter.update();
                if(flag == 1) {
                    cout << "Remove the bad set!" << endl;
                    calib.removeBadMeasurement();
                }
                else {
                    cout << "current measurement value:\n" << filter.current_measurement() << endl;
                    cout << "current mean value:\n" << filter.current_mean() << endl;
                    cout << "current covariance matrix:\n" << filter.current_covar() << endl;
                }
            }
        }
    } else{
        cout << "Wrong mode!\nPlease check input!" << endl;
    }
    string output_path = save_dir;
    filter.output(output_path);
    //filter.debug();
    return 0;
}
