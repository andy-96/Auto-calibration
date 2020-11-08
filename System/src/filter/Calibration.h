//
// Created by hanyunhai on 8/7/20.
//
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
#ifndef FILTER_CALIBRATION_H
#define FILTER_CALIBRATION_H
class calibration{
public:
    calibration() = default;
    calibration(vector<String> files, int windows, Size img_size, int total, int mode_):
        file_list(files), window_size(windows), image_size(img_size), max_index(total), mode(mode_) {
        cout << "Init successfully!" << endl;
    }
    void Calibration_camera(void);
    void update_points(string, string);
    vector<double> return_results(void);
    void debug(void);
    bool check_first_set(void);
    bool quit(void);
    void removeBadMeasurement(void);
private:
    int mode = 0;
    int runkernel(InputArray _m1, InputArray _m2, OutputArray _model);
    Mat homography_compu(InputArray _points1, InputArray _points2);
    void intrinsic_kernel(int);
    vector<String> file_list;
    int window_size = 0;
    int current_index = 0;
    int remove_index = 0;
    int incremental_index = 2;
    int max_index = 0;
    vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > image_points;
    double fx;
    double fy;
    Size image_size;
    vector<Mat> homography;
    };
#endif //FILTER_CALIBRATION_H
