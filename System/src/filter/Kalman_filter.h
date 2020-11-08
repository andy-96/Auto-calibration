//
// Created by hanyunhai on 8/6/20.
//
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <random>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
using namespace std;
using namespace cv;
using namespace Eigen;
#ifndef FILTER_KALMAN_FILTER_H
#define FILTER_KALMAN_FILTER_H
class kalman{
public:
    kalman(int mode_, int used_images, int scale): mode(mode_), current_images(used_images), scalar(scale) {}
    void init(double, double, double);
    void measurement(double, double);
    int update(void);
    void noise_set(double, double);
    void debug(void);
    Vector2d current_measurement(void);
    Vector2d current_mean(void);
    Matrix2d current_covar(void);
    void output(string);
    void set_groundtruth(double, double);
private:
    int mode = 0;
    int scalar = 0;
    int current_images = 0;
    Vector2d mean_value = Vector2d::Zero();
    Vector2d groundtruth = Vector2d::Zero();
    vector<Vector2d> errors;
    vector<Vector2d> mean_update;
    vector<Vector2d> measurement_list;
    vector<Vector2d> measurement_errors;
    Matrix2d covariance = Matrix2d::Identity();
    Matrix2d F = Matrix2d::Identity();
    Matrix2d H = Matrix2d::Identity();
    Vector2d new_measurement = Vector2d::Zero();
    Vector2d Currentinnovation = Vector2d::Zero();
    Vector2d Previousinnovation = Vector2d::Zero();
    Matrix2d system_noise = Matrix2d::Identity();
    Matrix2d measurement_noise = Matrix2d::Identity();
};
#endif //FILTER_KALMAN_FILTER_H
