//
//  filter.cpp
//
//
//  Created by Wing Man, Kwok on 31/1/2022.
//

#include <stdio.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int greyscale(cv::Mat &img, cv::Mat &grey_image ) {
    img.copyTo(grey_image);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols ; j++) {
            grey_image.at<Vec3b>(i,j)[0] = img.at<Vec3b>(i,j)[0];
            grey_image.at<Vec3b>(i,j)[1] = img.at<Vec3b>(i,j)[0];
            grey_image.at<Vec3b>(i,j)[2] = img.at<Vec3b>(i,j)[0];
        }
    }
    return 0;
}

int blur5x5( cv::Mat &img, cv::Mat &blur_16bit ){
    blur_16bit.create(img.rows, img.cols, CV_16SC3);
    img.convertTo(img, CV_16SC3);
    int value = 0;
    
    for (int i = 2; i < img.rows - 2; i++) {
        for (int j = 2; j < img.cols - 2 ; j++) {
            for (int c = 0; c<3; c++){
                value = img.at<Vec3s>(i-2,j-2)[c] + img.at<Vec3s>(i-2,j-1)[c] * 2  + img.at<Vec3s>(i-2,j)[c] * 4  + img.at<Vec3s>(i-2,j+1)[c] * 2  + img.at<Vec3s>(i-2,j+2)[c]  + img.at<Vec3s>(i-1,j-2)[c] * 2 + img.at<Vec3s>(i-1,j-1)[c] * 4 + img.at<Vec3s>(i-1,j)[c] * 8  + img.at<Vec3s>(i-1,j+1)[c] * 4  + img.at<Vec3s>(i-1,j+2)[c] * 2  + img.at<Vec3s>(i,j-2)[c] * 4  + img.at<Vec3s>(i,j-1)[c] * 8   + img.at<Vec3s>(i,j)[c] * 16  + img.at<Vec3s>(i,j+1)[c] * 8  + img.at<Vec3s>(i,j+2)[c] * 4  + img.at<Vec3s>(i+1,j-2)[c] * 2  + img.at<Vec3s>(i+1,j-1)[c] * 4   + img.at<Vec3s>(i+1,j)[c] * 8  + img.at<Vec3s>(i+1,j+1)[c] * 4  + img.at<Vec3s>(i+1,j+2)[c] * 2  + img.at<Vec3s>(i+2,j-2)[c] * 1  + img.at<Vec3s>(i+2,j-1)[c] * 2  + img.at<Vec3s>(i+2,j)[c] * 4  + img.at<Vec3s>(i+2,j+1)[c] * 2  + img.at<Vec3s>(i+2,j+2)[c];
                value = value / 100;
                blur_16bit.at<Vec3s>(i,j)[c] = value;
            }
        }
    }
    return 0;
}

int sobelX3x3(Mat &img, Mat &sobelx_16bit ) {
    int value=0;
    sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
    
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1 ; j++) {
            for (int c = 0; c<3; c++){
                value =  img.at<Vec3b>(i-1,j-1)[c] * -1  + img.at<Vec3b>(i-1,j)[c] * 0   + img.at<Vec3b>(i-1,j+1)[c] * 1 + img.at<Vec3b>(i,j-1)[c] * -2 + img.at<Vec3b>(i,j)[c] * 0 + img.at<Vec3b>(i,j+1)[c] * 2  + img.at<Vec3b>(i+1,j-1)[c] * -1 + img.at<Vec3b>(i+1,j)[c] * 0   + img.at<Vec3b>(i+1,j+1)[c] * 1;
                value = value / 4;
                sobelx_16bit.at<Vec3s>(i,j)[c] = value;
            }
        }
    }
    return 0;
}

int sobelY3x3(Mat &img, Mat &sobely_16bit ) {
    int value = 0;
    sobely_16bit.create(img.rows, img.cols, CV_16SC3);
    
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1 ; j++) {
            for (int c = 0; c<3; c++){
                value =  img.at<Vec3b>(i-1,j-1)[c] * 1  + img.at<Vec3b>(i-1,j)[c] * 2   + img.at<Vec3b>(i-1,j+1)[c] * 1 + img.at<Vec3b>(i,j-1)[c] * 0 + img.at<Vec3b>(i,j)[c] * 0 + img.at<Vec3b>(i,j+1)[c] * 0  + img.at<Vec3b>(i+1,j-1)[c] * -1 + img.at<Vec3b>(i+1,j)[c] * -2   + img.at<Vec3b>(i+1,j+1)[c] * -1;
                value = value / 4;
                sobely_16bit.at<Vec3s>(i,j)[c] = value;
            }
        }
    }
    return 0;
}

int magnitude(Mat &sobelx_16bit, Mat &sobely_16bit, Mat &gradientmag_16bit ) {
    int value = 0;
    gradientmag_16bit.create(sobelx_16bit.rows, sobelx_16bit.cols, CV_16SC3);
    
    for (int i = 1; i < sobelx_16bit.rows - 1; i++) {
        for (int j = 1; j < sobelx_16bit.cols - 1 ; j++) {
            for (int c = 0; c<3; c++){
                value =  sqrt(sobelx_16bit.at<Vec3s>(i,j)[c] * sobelx_16bit.at<Vec3s>(i,j)[c] + sobely_16bit.at<Vec3s>(i,j)[c] * sobely_16bit.at<Vec3s>(i,j)[c]);
                gradientmag_16bit.at<Vec3s>(i,j)[c] = value;
            }
        }
    }
    return 0;
}

int blurQuantize(Mat &blur_16bit, Mat &quantize_16bit, int levels) {
    
    quantize_16bit.create(blur_16bit.rows, blur_16bit.cols, CV_16SC3);
    int b = 255 / levels;
    int value = 0;
    
    for (int i = 1; i < blur_16bit.rows - 1; i++) {
        for (int j = 1; j < blur_16bit.cols - 1 ; j++) {
            for (int c = 0; c<3; c++){
                value =  blur_16bit.at<Vec3s>(i,j)[c] / b;
                value =  value * b;
                quantize_16bit.at<Vec3s>(i,j)[c] = value;
            }
        }
    }
    return 0;
}

int cartoon(Mat &quantize_16bit, Mat &gradientmag_16bit, Mat &cartoon_16bit, int levels, int magThreshold ) {
    
    cartoon_16bit.create(quantize_16bit.rows, quantize_16bit.cols, CV_16SC3);
    
    for (int i = 0; i < quantize_16bit.rows; i++) {
        for (int j = 0; j < quantize_16bit.cols ; j++) {
            for (int c = 0; c<3; c++){
                if (gradientmag_16bit.at<Vec3s>(i,j)[c] > magThreshold ){
                    cartoon_16bit.at<Vec3s>(i,j)[c] = 0;
                }
                else {
                    cartoon_16bit.at<Vec3s>(i,j)[c] = quantize_16bit.at<Vec3s>(i,j)[c];
                }
            }
        }
    }
    return 0;
}

int contrast(Mat &img, Mat &contrast_16bit ) {
    
    contrast_16bit.create(img.rows,img.cols, CV_16SC3);
    
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1 ; j++) {
            for (int c = 0; c<3; c++){
                contrast_16bit.at<Vec3s>(i,j)[c] = saturate_cast<uchar>( 1.2 * img.at<Vec3b>(i,j)[c] + 20 );
            }
        }
    }
    return 0;
}

int mirror(Mat &img, Mat &mirror_8bit ) {
    
    img.copyTo(mirror_8bit);
    
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            for (int c = 0; c<3; c++){
                mirror_8bit.at<Vec3b>(i,j)[c] = img.at<Vec3b>(i,img.cols -j)[c];
            }
        }
    }
    return 0;
}

int meme(Mat &img, string &text){
    cout << "What is meme of the day? \n";
    getline(cin, text);
    cout << text << "\n";
    putText(img, text, Point(50, 150), FONT_HERSHEY_DUPLEX, 1.5, Scalar(255,192,203),2,false);
    return 0;
}

int blending(Mat &img, Mat &dst) {
    double alpha = 0.5;
    double beta = ( 1.0 - alpha );
    Mat cny;
    
    cny = imread("cny.jpg");
    addWeighted( cny, alpha, img, beta, 0.0, dst);
    
    return 0;
}
