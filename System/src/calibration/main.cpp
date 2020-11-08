#include<iostream>
#include<string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <random>
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using namespace std;
using namespace cv;
vector<vector<cv::Point3f> > object_points;
vector<vector<cv::Point2f> > image_points;
cv::Mat intrinsic_matrix, distortion_coeffs;
cv::Mat intrinsic_matrix_dlt = Mat(3 , 3, CV_64FC1);
void Calibration_camera(vector<vector<Point3f> >, vector<vector<Point2f> >, Size, Mat&);
Mat intrinsic_kernel(vector<Mat>&, int nimages, Size&);
Mat homography_compu(InputArray, InputArray, int, double, int, double);
int runkernel(InputArray, InputArray, OutputArray);

int main(int argc, char** argv) {
    String intersection = String(argv[1]); //intersection_summary.txt
    String gt_dlt = String(argv[2]); //groundtruth for DLT method
    String gt_opt = String(argv[3]); //groundtruth for optimization
    String raw_img_dir = String(argv[4]); //raw_image_dir
    String imageP_dir = String(argv[5]); //image_points_dir
    String objectP_dir = String(argv[6]); //object_points_dir
    String save_dir = String(argv[7]); //store directory
    String img_format = String(argv[8]); //jpg or png
    //Groundtruth for the intrinsic matrix of my cellphone's camera
    vector<String> file_list;
    fstream outFile;
    char buffer[256];
    outFile.open(intersection, ios::in);
    while(!outFile.eof())
    {
        outFile.getline(buffer,256,'\n');//getline(char *,int,char)
        if(String(buffer).size() >= 1) {
            file_list.push_back(String(buffer));
            cout << String(buffer) << endl;
        }
    }
    if(file_list.size() <= 2){
        cout << "Not enough images for calibration!" << endl;
        return 0;
    }
    file_list.push_back(String("111"));
    outFile.close();
    cv::FileStorage fs1(gt_dlt, cv::FileStorage::READ);
    //fs1.open("intrinsics_dlt.xml", cv::FileStorage::READ);
    cv::FileStorage fs2(gt_opt, cv::FileStorage::READ);
    //fs2.open("intrinsics_opt.xml", cv::FileStorage::READ);
    cv::Mat intrinsic_matrix_gt_dlt, intrinsic_matrix_gt_opt, distortion_coeffs_gt_opt;
    fs1["camera_matrix"] >> intrinsic_matrix_gt_dlt;
    fs2["camera_matrix"] >> intrinsic_matrix_gt_opt;
    fs2["distortion_coefficients"] >> distortion_coeffs_gt_opt;
    cout << "Groundtruth using DLT only:\n";
    cout << intrinsic_matrix_gt_dlt << endl;
    cout << "Groundtruth using DLT and one-step LM optimization:\n";
    cout << intrinsic_matrix_gt_opt << endl;
    cout << distortion_coeffs_gt_opt << endl;
    double fx_opt_ = intrinsic_matrix_gt_opt.at<double>(0, 0);
    double fy_opt_ = intrinsic_matrix_gt_opt.at<double>(1, 1);
    double fx_dlt_ = intrinsic_matrix_gt_dlt.at<double>(0, 0);
    double fy_dlt_ = intrinsic_matrix_gt_dlt.at<double>(1, 1);
    vector<int> dataset;
    for(int i = 0; i != file_list.size() - 1; i++)
        dataset.push_back(i);
    cout << dataset.size() << endl;
    cout << "-----------------" << endl;
    int start = 0;
    vector<double> error_fx_DLT;
    vector<double> error_fy_DLT;
    vector<double> error_fx_Optim;
    vector<double> error_fy_Optim;
    vector<double> fx_DLT;
    vector<double> fy_DLT;
    vector<double> fx_Optim;
    vector<double> fy_Optim;
    for (int index_ = 0; index_ != dataset.size(); index_++) {
//            int index = dataset[index_];
//            int index_file = dataset[index_];
        int index = start;
        int index_file = index_;
        string raw_img_tmp(raw_img_dir);
        raw_img_tmp += file_list[0].substr(0, file_list[0].length() - 2);
        raw_img_tmp += String(".");
        raw_img_tmp += img_format;
        Mat img = imread(raw_img_tmp);
        Size img_size = img.size();
        cout << "REMOVED FILE NAME: " << file_list[dataset[index_]] << endl;
        while (true) {
            if(index != dataset[index_]) {
                string name = file_list[index];
                string file_name;
                string file_name2;
                file_name = imageP_dir + name + String("_image.txt"); //image points
                file_name2 = objectP_dir + name + String("_object.txt"); //object points
                ifstream ifs;
                // cout << file_name << endl;
                // cout << name << ", ";
                ifs.open(file_name, ios::in);
                if (!ifs.is_open()) {
                    //cout << "\n" << file_name << " dosen't exist!" << endl;
                    break;
                } else {
                    float x;
                    float y;
                    vector<Point2f> image_point;
                    while (!ifs.eof()) {
                        ifs >> x;
                        ifs >> y;
                        image_point.push_back(Point2f(x, y));
                    }
                    image_point.pop_back();
//                    if(index_ == dataset.size() - 1)
//                        for(auto point = image_point.begin(); point != image_point.end(); point++)
//                            cout << *point << endl;
                    image_point.pop_back();
                    ifs.close();
                    vector<cv::Point3f> known_object;
                    ifstream ofs;
                    ofs.open(file_name2, ios::in);
                    float x2, y2, z2;
                    while (!ofs.eof()) {
                        ofs >> x2;
                        ofs >> y2;
                        ofs >> z2;
                        known_object.push_back(Point3f(x2, y2, z2));
                    }
                    known_object.pop_back();
//                    if(index_ == dataset.size() - 1)
//                        for(auto point = known_object.begin(); point != known_object.end(); point++)
//                            cout << *point << endl;
                    known_object.pop_back();
                    ofs.close();
                    object_points.push_back(known_object);
                    image_points.push_back(image_point);
                    image_point.clear();
                    known_object.clear();
                    index++;
                }
            } else
                index++;
        }
        // cout << "END" << endl << endl;
        Calibration_camera(object_points, image_points, img_size, intrinsic_matrix_dlt);
        cout << intrinsic_matrix_dlt << endl;
//            cout << "Image size:" << img_size << endl;
        string file_name0(save_dir + String("Traffic_sign_auto/intrinsics_dlt") + to_string(index_file) + String(".xml"));
        cv::FileStorage fs0(file_name0, cv::FileStorage::WRITE);
        fs0 << "image_width" << img_size.width << "image_height" << img_size.height
           << "camera_matrix" << intrinsic_matrix_dlt;
        fs0.release();
//        double err = cv::calibrateCamera(object_points, image_points, img_size, intrinsic_matrix,
//                                         distortion_coeffs, cv::noArray(), cv::noArray(),
//                                         cv::CALIB_ZERO_TANGENT_DIST,
//                                         TermCriteria(
//                                                 TermCriteria::COUNT + TermCriteria::EPS, 1, DBL_EPSILON));
//        cout << "error:" << err << endl;
//            cout << "intrinsic matrix:" << intrinsic_matrix;
//            cout << "distortion coefficients: " << distortion_coeffs << endl;
//        error_fx_Optim.push_back((intrinsic_matrix.at<double>(0, 0) - fx_opt_) / fx_opt_);
//        error_fy_Optim.push_back((intrinsic_matrix.at<double>(1, 1) - fy_opt_) / fy_opt_);
//        fx_Optim.push_back(intrinsic_matrix.at<double>(0, 0));
//        fy_Optim.push_back(intrinsic_matrix.at<double>(1, 1));
        error_fx_DLT.push_back((intrinsic_matrix_dlt.at<double>(0, 0) - fx_dlt_) / fx_dlt_);
        error_fy_DLT.push_back((intrinsic_matrix_dlt.at<double>(1, 1) - fy_dlt_) / fy_dlt_);
        fx_DLT.push_back(intrinsic_matrix_dlt.at<double>(0, 0));
        fy_DLT.push_back(intrinsic_matrix_dlt.at<double>(1, 1));
//        string file_name(save_dir + String("Traffic_sign_auto/intrinsics_opt") + to_string(index_file) + String(".xml"));
//        cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
//        fs << "image_width" << img_size.width << "image_height" << img_size.height
//           << "camera_matrix" << intrinsic_matrix << "distortion_coefficients"
//           << distortion_coeffs;
//        fs.release();
//        cout << endl;
        cout << "Exp" << (index_+1) << ":" << endl;
        object_points.clear();
        image_points.clear();
    }
    string raw_img_tmp(raw_img_dir);
    raw_img_tmp += file_list[0].substr(0, file_list[0].length() - 2);
    raw_img_tmp += String(".");
    raw_img_tmp += img_format;
    Mat img = imread(raw_img_tmp);
    Size img_size = img.size();
    for(int index = 0; index != file_list.size() - 1; index++) {
        string name = file_list[index];
        string file_name;
        string file_name2;
        file_name = imageP_dir + name + String("_image.txt"); //image points
        file_name2 = objectP_dir + name + String("_object.txt"); //object points
        ifstream ifs;
        ifs.open(file_name, ios::in);
        float x;
        float y;
        vector<Point2f> image_point;
        while (!ifs.eof()) {
            ifs >> x;
            ifs >> y;
            image_point.push_back(Point2f(x, y));
        }
        image_point.pop_back();
        image_point.pop_back();
        ifs.close();
        vector<cv::Point3f> known_object;
        ifstream ofs;
        ofs.open(file_name2, ios::in);
        float x2, y2, z2;
        while (!ofs.eof()) {
            ofs >> x2;
            ofs >> y2;
            ofs >> z2;
            known_object.push_back(Point3f(x2, y2, z2));
        }
        known_object.pop_back();
        known_object.pop_back();
        ofs.close();
        object_points.push_back(known_object);
        image_points.push_back(image_point);
        image_point.clear();
        known_object.clear();
    }
    Calibration_camera(object_points, image_points, img_size, intrinsic_matrix_dlt);
    cout << intrinsic_matrix_dlt << endl;
//            cout << "Image size:" << img_size << endl;
    string file_name0(save_dir + String("Traffic_sign_auto/intrinsics_dlt") + to_string(-1) + String(".xml"));
    cv::FileStorage fs0(file_name0, cv::FileStorage::WRITE);
    fs0 << "image_width" << img_size.width << "image_height" << img_size.height
        << "camera_matrix" << intrinsic_matrix_dlt;
//    fs0.release();
//    double err = cv::calibrateCamera(object_points, image_points, img_size, intrinsic_matrix,
//                                     distortion_coeffs, cv::noArray(), cv::noArray(),
//                                     cv::CALIB_ZERO_TANGENT_DIST,
//                                     TermCriteria(
//                                             TermCriteria::COUNT + TermCriteria::EPS, 10, DBL_EPSILON));
//    cout << "error:" << err << endl;
//            cout << "intrinsic matrix:" << intrinsic_matrix;
//            cout << "distortion coefficients: " << distortion_coeffs << endl;
//    error_fx_Optim.push_back((intrinsic_matrix.at<double>(0, 0) - fx_opt_) / fx_opt_);
//    error_fy_Optim.push_back((intrinsic_matrix.at<double>(1, 1) - fy_opt_) / fy_opt_);
//    fx_Optim.push_back(intrinsic_matrix.at<double>(0, 0));
//    fy_Optim.push_back(intrinsic_matrix.at<double>(1, 1));
    error_fx_DLT.push_back((intrinsic_matrix_dlt.at<double>(0, 0) - fx_dlt_) / fx_dlt_);
    error_fy_DLT.push_back((intrinsic_matrix_dlt.at<double>(1, 1) - fy_dlt_) / fy_dlt_);
    fx_DLT.push_back(intrinsic_matrix_dlt.at<double>(0, 0));
    fy_DLT.push_back(intrinsic_matrix_dlt.at<double>(1, 1));
//    string file_name(save_dir + String("Traffic_sign_auto/intrinsics_opt") + to_string(-1) + String(".xml"));
//    cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
//    fs << "image_width" << img_size.width << "image_height" << img_size.height
//       << "camera_matrix" << intrinsic_matrix << "distortion_coefficients"
//       << distortion_coeffs;
//    fs.release();
    object_points.clear();
    image_points.clear();
    string fx_dlt = save_dir + String("fx_DLT.txt");
    ofstream _file1(fx_dlt);
    string fy_dlt = save_dir + String("fy_DLT.txt");
    ofstream _file2(fy_dlt);
    string error_fx_dlt = save_dir + String("error_fx_DLT.txt");
    ofstream _file3(error_fx_dlt);
    string error_fy_dlt = save_dir + String("error_fy_DLT.txt");
    ofstream _file4(error_fy_dlt);
//    string fx_optim = save_dir + String("fx_Optim.txt");
//    ofstream _file5(fx_optim);
//    string fy_optim = save_dir + String("fy_Optim.txt");
//    ofstream _file6(fy_optim);
//    string error_fx_optim = save_dir + String("error_fx_Optim.txt");
//    ofstream _file7(error_fx_optim);
//    string error_fy_optim = save_dir + String("error_fy_Optim.txt");
//    ofstream _file8(error_fy_optim);
    for(int i = 0; i != error_fx_DLT.size(); i++){
        _file1 << fx_DLT[i] << endl;
        _file2 << fy_DLT[i] << endl;
        _file3 << error_fx_DLT[i] << endl;
        _file4 << error_fy_DLT[i] << endl;
//        _file5 << fx_Optim[i] << endl;
//        _file6 << fy_Optim[i] << endl;
//        _file7 << error_fx_Optim[i] << endl;
//        _file8 << error_fy_Optim[i] << endl;
    }
    string fx_gt = save_dir + String("groundtruth.txt");
    ofstream _file0(fx_gt);
    _file0 << fx_dlt_ << endl << fy_dlt_ << endl << fx_opt_ << endl << fy_opt_ << endl;
    return 0;
}

int runkernel(InputArray _m1, InputArray _m2, OutputArray _model){
    Mat m1 = _m1.getMat(), m2 = _m2.getMat();
    int i, count = m1.checkVector(2);
    const Point2f* M = m1.ptr<Point2f>();
    const Point2f* m = m2.ptr<Point2f>();

    double LtL[9][9], W[9][1], V[9][9];
    Mat _LtL( 9, 9, CV_64F, &LtL[0][0] );
    Mat matW( 9, 1, CV_64F, W );
    Mat matV( 9, 9, CV_64F, V );
    Mat _H0( 3, 3, CV_64F, V[8] );
    Mat _Htemp( 3, 3, CV_64F, V[7] );
    Point2d cM(0,0), cm(0,0), sM(0,0), sm(0,0);

    for( i = 0; i < count; i++ )
    {
        cm.x += m[i].x; cm.y += m[i].y;
        cM.x += M[i].x; cM.y += M[i].y;
    }

    cm.x /= count;
    cm.y /= count;
    cM.x /= count;
    cM.y /= count;

    for( i = 0; i < count; i++ )
    {
        sm.x += fabs(m[i].x - cm.x);
        sm.y += fabs(m[i].y - cm.y);
        sM.x += fabs(M[i].x - cM.x);
        sM.y += fabs(M[i].y - cM.y); //absolute value
    }

    if( fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
        fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON )
        return 0;
    sm.x = count/sm.x; sm.y = count/sm.y;
    sM.x = count/sM.x; sM.y = count/sM.y; //Here, it used absolute value to quantify the pixel deviation of all points
    //insteard standard deviation.
    //other parts are correct as described in papers.

    double invHnorm[9] = { 1./sm.x, 0, cm.x, 0, 1./sm.y, cm.y, 0, 0, 1 };
    double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };
    //two transform matrix, the second is the inverse of the transform matrix.
    Mat _invHnorm( 3, 3, CV_64FC1, invHnorm );
    Mat _Hnorm2( 3, 3, CV_64FC1, Hnorm2 );

    _LtL.setTo(Scalar::all(0));
    for( i = 0; i < count; i++ )
    {
        double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
        double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
        double Lx[] = { X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x };
        double Ly[] = { 0, 0, 0, X, Y, 1, -y*X, -y*Y, -y };
        int j, k;
        for( j = 0; j < 9; j++ )
            for( k = j; k < 9; k++ )
                LtL[j][k] += Lx[j]*Lx[k] + Ly[j]*Ly[k]; // m^T*m
    }
    completeSymm( _LtL ); //m^Tm is a symmetric matrix, so this is right.

    eigen( _LtL, matW, matV ); //From this, I can tell that In OpenCV, it doesn't use svd but eigenvalues
    //so m should be A=m   m*9(for svd), ATA=mTm(9*9) for eigenvalue decomposition.
    //eigenvalue decomposition
    _Htemp = _invHnorm*_H0;
    _H0 = _Htemp*_Hnorm2;
    _H0.convertTo(_model, _H0.type(), 1./_H0.at<double>(2,2) );

    return 1;
}
Mat homography_compu(InputArray _points1, InputArray _points2, int method = 0,
                                  double ransacReprojThreshold = 3, int maxIters = 2000, double confidence = 0.995) {
    const double defaultRANSACReprojThreshold = 3;
    bool result = false;
    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    Mat src, dst, H, tempMask;
    int npoints = -1;
    for( int i = 1; i <= 2; i++ )
    {
        Mat& p = i == 1 ? points1 : points2;
        Mat& m = i == 1 ? src : dst;
        npoints = p.checkVector(2, -1, false);
        if( npoints < 0 )
        {
            npoints = p.checkVector(3, -1, false);
            convertPointsFromHomogeneous(p, p);
        }
        p.reshape(2, npoints).convertTo(m, CV_32F);
    }
    if( ransacReprojThreshold <= 0 )
        ransacReprojThreshold = defaultRANSACReprojThreshold;
    result = runkernel(src, dst, H) > 0;
    if( result && npoints > 4)
    {
        if( npoints > 0 )
        {
            Mat src1 = src.rowRange(0, npoints);
            Mat dst1 = dst.rowRange(0, npoints);
            src = src1;
            dst = dst1;
            Mat H8(8, 1, CV_64F, H.ptr<double>());
//            createLMSolver(makePtr<HomographyRefineCallback>(src, dst), 10)->run(H8);
//          LM_optimization(findhomogrphy)
        }
    }
    return H;
}
Mat intrinsic_kernel(vector<Mat>& homography, int nimages, Size& imageSize) {
    Ptr<CvMat> matA, _b, _allH;
    int i, j;
    double a[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    double H[9] = {0}, f[2] = {0};
    CvMat _f = cvMat( 2, 1, CV_64F, f );
    matA.reset(cvCreateMat( 2*nimages, 2, CV_64F ));
    _b.reset(cvCreateMat( 2*nimages, 1, CV_64F ));
    _allH.reset(cvCreateMat( nimages, 9, CV_64F ));
    a[2] = (!imageSize.width) ? 0.5 : (imageSize.width)*0.5;
    a[5] = (!imageSize.height) ? 0.5 : (imageSize.height)*0.5;
    for(int i = 0;i != nimages; i++){
        double *Ap = matA->data.db + i * 4;
        double *bp = _b->data.db + i * 2;
        double h[3], v[3], d1[3], d2[3];
        double n[4] = {0,0,0,0};
        Mat matH = homography[i];
        H[0] = matH.at<double>(0,0);
        H[1] = matH.at<double>(0,1);
        H[2] = matH.at<double>(0,2);
        H[3] = matH.at<double>(1,0);
        H[4] = matH.at<double>(1,1);
        H[5] = matH.at<double>(1,2);
        H[6] = matH.at<double>(2,0);
        H[7] = matH.at<double>(2,1);
        //H[6] = 0;
        //H[7] = 0;
        H[8] = matH.at<double>(2,2);
        //cout << matH << endl;
        memcpy( _allH->data.db + i*9, H, sizeof(H) );
        H[0] -= H[6]*a[2]; H[1] -= H[7]*a[2]; H[2] -= H[8]*a[2]; //After this operation:
        //H[0] = fx*R00, H[1] = fx*R01, H[2] = fx*t0
        H[3] -= H[6]*a[5]; H[4] -= H[7]*a[5]; H[5] -= H[8]*a[5];
        //H[3] = fy*R10, H[4] = fy*R11, H[5] = fy*t1
        //t0 t1 up to a scale

        //H[6] = R20, H[7] = R21  H[8] = t2
        //a[2] -> cx
        //a[5] -> cy
        for( j = 0; j < 3; j++ )
        {
            double t0 = H[j*3], t1 = H[j*3+1];
            h[j] = t0; v[j] = t1;
            d1[j] = (t0 + t1)*0.5;
            d2[j] = (t0 - t1)*0.5;
            n[0] += t0*t0; n[1] += t1*t1; //the initial value of n is zero
            n[2] += d1[j]*d1[j]; n[3] += d2[j]*d2[j];
        }
        for( j = 0; j < 4; j++ )
            n[j] = 1./std::sqrt(n[j]);
        for( j = 0; j < 3; j++ )
        {
            h[j] *= n[0]; v[j] *= n[1];
            d1[j] *= n[2]; d2[j] *= n[3];
        }
        Ap[0] = h[0]*v[0]; Ap[1] = h[1]*v[1];  //Ap -> matA
        Ap[2] = d1[0]*d2[0]; Ap[3] = d1[1]*d2[1];  //bp -> _b
        //Ap
        bp[0] = -h[2]*v[2]; bp[1] = -d1[2]*d2[2];  //remove scale?
    }
    cvSolve( matA, _b, &_f, CV_NORMAL + CV_SVD ); //Indeed, there are only two parameters in _f -> f_x,f_y
    a[0] = std::sqrt(fabs(1./f[0]));
    a[4] = std::sqrt(fabs(1./f[1]));
    Mat cameraMatrix = Mat(3 , 3, CV_64FC1);
    cameraMatrix.at<double>(0,0) = a[0];
    cameraMatrix.at<double>(0,2) = a[2];
    cameraMatrix.at<double>(1,1) = a[4];
    cameraMatrix.at<double>(1,2) = a[5];
    cameraMatrix.at<double>(2,2) = 1;
    cameraMatrix.at<double>(0,1) = 0;
    cameraMatrix.at<double>(1,0) = 0;
    cameraMatrix.at<double>(2,0) = 0;
    cameraMatrix.at<double>(2,1) = 0;
    return cameraMatrix;
}
void Calibration_camera(vector<vector<Point3f> > objectPoints, vector<vector<Point2f> > imagePoints,
                                     Size imageSize, Mat& cameraMatrix) {

    int i, nimages;
    nimages = objectPoints.size();
    vector<Mat> homography_;
    for(i = 0; i < nimages; i++) {
        //homography_.push_back(findHomography(objectPoints[i], imagePoints[i]));
        homography_.push_back(homography_compu(objectPoints[i], imagePoints[i]));
        //cout << "-----------------------------" << endl;
        //cout << homography_compu(objectPoints[i], imagePoints[i]) << endl;
        //cout << "-----------------------------" << endl;
        //in Opencv  built-in function findHomography seems to use LM solver
        //homography_compu dont use LM solver(it only uses linear method)
        //instead of algebraic error(homography_compu), use geometric error(findHomography)
        //from multi-view geometry, it should be considered as the starting point.
    }
    cameraMatrix = intrinsic_kernel(homography_, nimages, imageSize);
    //camera intrinsic matrix parameter.
}
