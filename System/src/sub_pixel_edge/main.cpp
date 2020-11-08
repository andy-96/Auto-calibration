#include "edgeTest.h"
#include "WriteFile.h"
using namespace std;
using namespace cv;
const int max_value_H = 360/2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
vector<int> parameter1 = {0, 93, 16};
vector<int> parameter2 = {0, 112, 51};
int high_H = max_value_H, high_S = max_value, high_V = max_value;
int main(int argc, char** argv)
{
    String file_path = String(argv[1]); //boxes.txt
    String image_dir = String(argv[2]); //image_dir for example: /box_imgs/cell_phone/
    String useful_dir = String(argv[3]); // useful.txt
    String out_dir = String(argv[4]); //output files
    String img_format = String(argv[5]); //jpg or png
    Mat grayImage, dstImage, srcImage, HSVImage;
    vector<String> file_list;
    char buffer[256];
    fstream outFile;
    outFile.open(file_path, ios::in);
    while(!outFile.eof())
    {
        outFile.getline(buffer,256,'\n');//getline(char *,int,char)
        file_list.push_back(String(buffer));
    }
    outFile.close();
    //cout << file_list[0].substr(0, file_list[0].length() - 4);
    //cout << file_list[0] << file_list[file_list.size() - 2] << endl;
    ofstream mask_out(useful_dir);
    for(int op = 0; op != file_list.size() && file_list[op].size() >= 4; op++) {
        String file_name = file_list[op];
        String image_dir_tmp(image_dir);
        image_dir_tmp += file_name;
        image_dir_tmp += String("box.");
        image_dir_tmp += img_format;
        srcImage = imread(image_dir_tmp);
        if (srcImage.empty()) {
            cout << "load error!" << endl;
            return -1;
        }
        //parameters setting
        double *x;          /* x[n] y[n] coordinates of result contour point n */
        double *y;
        double *theta;
        int *curve_limits;  /* limits of the curves in the x[] and y[] */
        int N, M;         /* result: N contour points, forming M curves */
        double S = 0; /* default sigma=0 */
        double H = 15; /* default th_h=0  */
        double L = 5; /* default th_l=0  */
        double W = 1; /* default W=1.3   */
        //String tmp = file_name.substr(0, file_name.length() - 9);
        String tmp_pdf = out_dir + file_name + String("_output.pdf");
        String tmp_txt1 = out_dir + file_name + String("_inner_polygon.txt");
        String tmp_txt3 = out_dir + file_name + String("_inner_T.txt");
        const char *pdf_out = tmp_pdf.c_str();    /*pdf filename  -> image*/
        const char *txt_out1 = tmp_txt1.c_str();  /*txt filename  -> pixel position(sub-pixel accuracy)*/
        const char *txt_out3 = tmp_txt3.c_str();
        cvtColor(srcImage, HSVImage, COLOR_BGR2HSV);
        inRange(HSVImage, Scalar(parameter1[0], parameter1[1], parameter1[2]), Scalar(high_H, high_S, high_V), grayImage);
        imshow(window_capture_name, srcImage); //TODO
        imshow(window_detection_name, grayImage);
        waitKey(3);
        //equalizeHist(grayImage, grayImage);
        //cvtColor(refImage, GrayrefImage, COLOR_BGR2GRAY);
        dstImage = grayImage;
        const int iHeight = dstImage.rows;
        const int iWidth = dstImage.cols;
        uchar *pSrc = grayImage.data;//new uchar[iHeight*iWidth];
        uchar *pDst = dstImage.data;
        //imshow("input image", grayImage);
        devernay(&x, &y, &N, &theta, &curve_limits, &M, pSrc, pDst, iWidth, iHeight, S, H, L);
        bool valid_ = false;
        int T_index, inner_polygon_index;
        if (pdf_out != NULL)
            valid_ = write_curves_pdf(x, y, curve_limits, M, pdf_out, iWidth, iHeight, W, T_index, inner_polygon_index);
        cout << ">> " << file_name << "box." << img_format << ":" << valid_ << endl;
        if(valid_ == true) {
            int work;
            if (txt_out1 != NULL)
                work = write_curves_txt_costomized(x, y, theta, curve_limits, M, txt_out1, txt_out3, T_index, inner_polygon_index);
            if(work == 2)
                mask_out << file_name << endl;
        } //TODO
    }
    mask_out.close();
    //system("pause");
    return 0;
}
