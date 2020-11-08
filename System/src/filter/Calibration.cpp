//
// Created by hanyunhai on 8/7/20.
//
#include "Calibration.h"
using namespace std;
using namespace cv;
bool calibration::check_first_set(void){
    if(current_index == 0)
        return true;
    else
        return false;
}
bool calibration::quit(void){
    if(current_index == max_index + 1 - window_size)
        return true;
    else
        return false;
}
vector<double> calibration::return_results(void){
    return vector<double>{fx, fy};
}
int calibration::runkernel(InputArray _m1, InputArray _m2, OutputArray _model){
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
Mat calibration::homography_compu(InputArray _points1, InputArray _points2) {
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
void calibration::Calibration_camera(void) {
    int i;
    int nimages;
    if(mode == 1)
        nimages = file_list.size() - 1;
    else if(mode == 2)
        nimages = window_size;
    else if(mode == 3)
        nimages = incremental_index - 1;
    vector<Mat> homography_;
    for(i = 0; i < nimages; i++) {
        //homography_.push_back(findHomography(objectPoints[i], imagePoints[i]));
        homography.push_back(homography_compu(object_points[i], image_points[i]));
//        cout << "-----------------------------" << endl;
//        cout << homography_compu(object_points[i], image_points[i]) << endl;
//        cout << "-----------------------------" << endl;
        //in Opencv  built-in function findHomography seems to use LM solver
        //homography_compu dont use LM solver(it only uses linear method)
        //instead of algebraic error(homography_compu), use geometric error(findHomography)
        //from multi-view geometry, it should be considered as the starting point.
    }
    intrinsic_kernel(nimages);
    //camera intrinsic matrix parameter.
}
void calibration::intrinsic_kernel(int n1) {
    int nimages = n1;
    Size imageSize = image_size;
    Ptr<CvMat> matA, _b, _allH;
    int i, j;
    double a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1};
    double H[9] = {0}, f[2] = {0};
    CvMat _f = cvMat(2, 1, CV_64F, f);
    matA.reset(cvCreateMat(2 * nimages, 2, CV_64F));
    _b.reset(cvCreateMat(2 * nimages, 1, CV_64F));
    _allH.reset(cvCreateMat(nimages, 9, CV_64F));
    a[2] = (!imageSize.width) ? 0.5 : (imageSize.width) * 0.5;
    a[5] = (!imageSize.height) ? 0.5 : (imageSize.height) * 0.5;
    for (int i = 0; i != nimages; i++) {
        double *Ap = matA->data.db + i * 4;
        double *bp = _b->data.db + i * 2;
        double h[3], v[3], d1[3], d2[3];
        double n[4] = {0, 0, 0, 0};
        Mat matH = homography[i];
        H[0] = matH.at<double>(0, 0);
        H[1] = matH.at<double>(0, 1);
        H[2] = matH.at<double>(0, 2);
        H[3] = matH.at<double>(1, 0);
        H[4] = matH.at<double>(1, 1);
        H[5] = matH.at<double>(1, 2);
        H[6] = matH.at<double>(2, 0);
        H[7] = matH.at<double>(2, 1);
        //H[6] = 0;
        //H[7] = 0;
        H[8] = matH.at<double>(2, 2);
        //cout << matH << endl;
        memcpy(_allH->data.db + i * 9, H, sizeof(H));
        H[0] -= H[6] * a[2];
        H[1] -= H[7] * a[2];
        H[2] -= H[8] * a[2]; //After this operation:
        //H[0] = fx*R00, H[1] = fx*R01, H[2] = fx*t0
        H[3] -= H[6] * a[5];
        H[4] -= H[7] * a[5];
        H[5] -= H[8] * a[5];
        //H[3] = fy*R10, H[4] = fy*R11, H[5] = fy*t1
        //t0 t1 up to a scale

        //H[6] = R20, H[7] = R21  H[8] = t2
        //a[2] -> cx
        //a[5] -> cy
        for (j = 0; j < 3; j++) {
            double t0 = H[j * 3], t1 = H[j * 3 + 1];
            h[j] = t0;
            v[j] = t1;
            d1[j] = (t0 + t1) * 0.5;
            d2[j] = (t0 - t1) * 0.5;
            n[0] += t0 * t0;
            n[1] += t1 * t1; //the initial value of n is zero
            n[2] += d1[j] * d1[j];
            n[3] += d2[j] * d2[j];
        }
        for (j = 0; j < 4; j++)
            n[j] = 1. / std::sqrt(n[j]);
        for (j = 0; j < 3; j++) {
            h[j] *= n[0];
            v[j] *= n[1];
            d1[j] *= n[2];
            d2[j] *= n[3];
        }
        Ap[0] = h[0] * v[0];
        Ap[1] = h[1] * v[1];  //Ap -> matA
        Ap[2] = d1[0] * d2[0];
        Ap[3] = d1[1] * d2[1];  //bp -> _b
        //Ap
        bp[0] = -h[2] * v[2];
        bp[1] = -d1[2] * d2[2];  //remove scale?
    }
    cvSolve(matA, _b, &_f, CV_NORMAL + CV_SVD); //Indeed, there are only two parameters in _f -> f_x,f_y
    a[0] = std::sqrt(fabs(1. / f[0]));
    fx = a[0];
    a[4] = std::sqrt(fabs(1. / f[1]));
    fy = a[4];
    homography.clear();
}
void calibration::update_points(string imageP, string objectP){
    object_points.clear();
    image_points.clear();
    string file_path1 = imageP;
    string file_path2 = objectP;
    //mode1 -> N-1 images out of N sets  mode2 -> moving window mode3 -> incremental
    if(mode == 1)
    {
        cout << endl << "REMOVED FILE NAME: " << file_list[remove_index] << endl;
        for(int j = 0; j != file_list.size(); j++){
            if(j != remove_index) {
                string file_name = file_list[j];
                // cout << "file_name:" << file_name << endl;
                // cout << file_name << ", ";
                string file_name1, file_name2;
                file_name1 = file_path1 + file_name + string("_image.txt");
                file_name2 = file_path2 + file_name + string("_object.txt");
                ifstream ifs;
                ifs.open(file_name1, ios::in);
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
                image_points.push_back(image_point);
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
                image_point.clear();
                known_object.clear();
            }
        }
        remove_index++;
        // cout << "END" << endl;
    }
    else if(mode == 2) {
        cout << endl << "FILE NAMES: ";
        for (int index = current_index; index < current_index + window_size; index++) {
            string file_name = file_list[index];
            // cout << "file_name:" << file_name << endl;
            cout << file_name << ", ";
            string file_name1, file_name2;
            file_name1 = file_path1 + file_name + string("_image.txt");
            file_name2 = file_path2 + file_name + string("_object.txt");
            ifstream ifs;
            ifs.open(file_name1, ios::in);
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
            image_points.push_back(image_point);
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
            image_point.clear();
            known_object.clear();
        }
        current_index++;
        cout << "END" << endl;
    }
    else if(mode == 3){
        if (incremental_index > 0) cout << endl << "ADDED FILE NAME:" << file_list[incremental_index-1] << endl;
        for(int index = 0; index < incremental_index; index++){
            string file_name = file_list[index];
            // cout << "file_name:" << file_name << endl;
            // cout << file_name << ", ";
            string file_name1, file_name2;
            file_name1 = file_path1 + file_name + string("_image.txt");
            file_name2 = file_path2 + file_name + string("_object.txt");
            ifstream ifs;
            ifs.open(file_name1, ios::in);
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
            image_points.push_back(image_point);
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
            image_point.clear();
            known_object.clear();
        }
        incremental_index++;
        // cout << "END" << endl;
    }
    else{
        cout << "Wrong mode!\nPlease check input!" << endl;
    }
}
void calibration::debug(void){
    cout << "current_index:" << current_index << endl;
    cout << "window_size:" << window_size << endl;
    cout << "max_index:" << max_index << endl;
    for(int i = current_index; i < max_index; i++)
        cout << file_list[i] << endl;
}
void calibration::removeBadMeasurement(void){
    incremental_index--;
    file_list.erase(file_list.begin() + incremental_index - 1);
}