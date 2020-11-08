//
// Created by hanyunhai on 8/6/20.
//
#include "Kalman_filter.h"
using namespace std;
using namespace cv;
using namespace Eigen;
void kalman::debug(void){
    cout << "mean_value:\n" << mean_value << endl;
    cout << "covariance:\n" << covariance << endl;
    cout << "F:\n" << F << endl;
    cout << "H:\n" << H << endl;
    cout << "new_measurement:\n" << new_measurement << endl;
    cout << "system_noise:\n" << system_noise << endl;
    cout << "measurement_noise:\n" << measurement_noise << endl;
}
void kalman::noise_set(double epsilon, double delta){
    system_noise *= epsilon;
    measurement_noise *= delta;
}
void kalman::output(string output_path){
    string fx_dlt = output_path + string("fx_kalman.txt");
    ofstream _file1(fx_dlt);
    string fy_dlt = output_path + string("fy_kalman.txt");
    ofstream _file2(fy_dlt);
    string error_fx_dlt = output_path + string("fx_kalman_error.txt");
    ofstream _file3(error_fx_dlt);
    string error_fy_dlt = output_path + string("fy_kalman_error.txt");
    ofstream _file4(error_fy_dlt);
    string measurement_fx = output_path + string("fx_measurement.txt");
    ofstream _file5(measurement_fx);
    string measurement_fy = output_path + string("fy_measurement.txt");
    ofstream _file6(measurement_fy);
    string error_fx_mea = output_path + string("fx_measurement_error.txt");
    ofstream _file7(error_fx_mea);
    string error_fy_mea = output_path + string("fy_measurement_error.txt");
    ofstream _file8(error_fy_mea);
    for(int i = 0; i != mean_update.size(); i++){
        _file1 << mean_update[i](0,0) << endl;
        _file2 << mean_update[i](1,0) << endl;
        _file3 << errors[i](0,0) << endl;
        _file4 << errors[i](1,0) << endl;
        _file5 << measurement_list[i](0,0) << endl;
        _file6 << measurement_list[i](1,0) << endl;
        _file7 << measurement_errors[i](0,0) << endl;
        _file8 << measurement_errors[i](1,0) << endl;
    }
    string gt = output_path + String("kalman_groundtruth.txt");
    ofstream _file0(gt);
    _file0 << groundtruth(0,0) << endl << groundtruth(1,0) << endl;
}
void kalman::init(double fx, double fy, double cova_init){
    mean_value(0,0) = fx;
    mean_value(1,0) = fy;
    new_measurement(0,0) = fx;
    new_measurement(1,0) = fy;
    covariance *= cova_init;
    mean_update.push_back(mean_value);
    double fx_error = (mean_value(0,0) - groundtruth(0,0)) / groundtruth(0,0);
    double fy_error = (mean_value(1,0) - groundtruth(1,0)) / groundtruth(1,0);
    Vector2d error;
    error << fx_error, fy_error;
    errors.push_back(error);
    measurement_list.push_back(new_measurement);
    measurement_errors.push_back(error);
}
void kalman::set_groundtruth(double fx_dlt, double fy_dlt){
    groundtruth(0,0) = fx_dlt;
    groundtruth(1,0) = fy_dlt;
}
void kalman::measurement(double new_fx, double new_fy){
    new_measurement(0,0) = new_fx;
    new_measurement(1,0) = new_fy;
}
Vector2d kalman::current_measurement(void){
    return new_measurement;
}
Vector2d kalman::current_mean(void){
    return mean_value;
}
Matrix2d kalman::current_covar(void) {
    return covariance;
}
int kalman::update(void){
    Matrix2d measurement_noise_tmp;
    if(mode == 1)
        measurement_noise_tmp = measurement_noise / current_images;
    else if(mode == 2)
        measurement_noise_tmp = measurement_noise / current_images;
    else{
        measurement_noise_tmp = measurement_noise / current_images;
        current_images += 1;
    }
    Vector2d mean_hat = F * mean_value;
    Matrix2d covariance_hat = F * covariance * F.transpose() + system_noise;
    Matrix2d tmp = H * covariance_hat * H.transpose() + measurement_noise_tmp;
    Matrix2d Kt = covariance_hat * H.transpose() * tmp.inverse();
    if(mode != 3){
        mean_value = mean_hat + Kt * (new_measurement - H * mean_hat);
        //cout << "innovation term:\n" << (new_measurement - H * mean_hat) << endl;
        covariance = (Matrix2d::Identity() - Kt * H) * covariance_hat;
        mean_update.push_back(mean_value);
        double fx_error = (mean_value(0,0) - groundtruth(0,0)) / groundtruth(0,0);
        double fy_error = (mean_value(1,0) - groundtruth(1,0)) / groundtruth(1,0);
        Vector2d error;
        error << fx_error, fy_error;
        errors.push_back(error);
        measurement_list.push_back(new_measurement);
        double fx_error_ = (new_measurement(0,0) - groundtruth(0,0)) / groundtruth(0,0);
        double fy_error_ = (new_measurement(1,0) - groundtruth(1,0)) / groundtruth(1,0);
        Vector2d error_;
        error_ << fx_error_, fy_error_;
        measurement_errors.push_back(error_);
        return 0;
    }
    else {
        if (Previousinnovation == Vector2d::Zero())
            Previousinnovation = new_measurement - H * mean_hat;
        else if (Currentinnovation == Vector2d::Zero())
            Currentinnovation = new_measurement - H * mean_hat;
        else {
            Previousinnovation = Currentinnovation;
            Currentinnovation = new_measurement - H * mean_hat;
        }
        if (Currentinnovation != Vector2d::Zero()) {
            double current = abs(Currentinnovation(0, 0)) + abs(Currentinnovation(1, 0));
            double previous = abs(Previousinnovation(0, 0)) + abs(Previousinnovation(1, 0));
            if (false) {
                Currentinnovation = Previousinnovation;
                current_images -= 1;
                return 1;
            } else {
                mean_value = mean_hat + Kt * (new_measurement - H * mean_hat);
                //cout << "innovation term:\n" << (new_measurement - H * mean_hat) << endl;
                covariance = (Matrix2d::Identity() - Kt * H) * covariance_hat;
                mean_update.push_back(mean_value);
                double fx_error = (mean_value(0, 0) - groundtruth(0, 0)) / groundtruth(0, 0);
                double fy_error = (mean_value(1, 0) - groundtruth(1, 0)) / groundtruth(1, 0);
                Vector2d error;
                error << fx_error, fy_error;
                errors.push_back(error);
                measurement_list.push_back(new_measurement);
                double fx_error_ = (new_measurement(0, 0) - groundtruth(0, 0)) / groundtruth(0, 0);
                double fy_error_ = (new_measurement(1, 0) - groundtruth(1, 0)) / groundtruth(1, 0);
                Vector2d error_;
                error_ << fx_error_, fy_error_;
                measurement_errors.push_back(error_);
                return 0;
            }
        } else {
            mean_value = mean_hat + Kt * (new_measurement - H * mean_hat);
            //cout << "innovation term:\n" << (new_measurement - H * mean_hat) << endl;
            covariance = (Matrix2d::Identity() - Kt * H) * covariance_hat;
            mean_update.push_back(mean_value);
            double fx_error = (mean_value(0, 0) - groundtruth(0, 0)) / groundtruth(0, 0);
            double fy_error = (mean_value(1, 0) - groundtruth(1, 0)) / groundtruth(1, 0);
            Vector2d error;
            error << fx_error, fy_error;
            errors.push_back(error);
            measurement_list.push_back(new_measurement);
            double fx_error_ = (new_measurement(0, 0) - groundtruth(0, 0)) / groundtruth(0, 0);
            double fy_error_ = (new_measurement(1, 0) - groundtruth(1, 0)) / groundtruth(1, 0);
            Vector2d error_;
            error_ << fx_error_, fy_error_;
            measurement_errors.push_back(error_);
            return 0;
        }
    }
}
