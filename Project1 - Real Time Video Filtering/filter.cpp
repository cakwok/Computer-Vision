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

int greyscale( cv::Mat &img, cv::Mat &grey_image ) {
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
    //img.copyTo(dst);
    //dst.convertTo(dst, CV_16SC3);
    //dst.create(img.size, CV_16SC3);
    
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1 ; j++) {
            for (int c = 0; c<3; c++){
                value =  img.at<Vec3b>(i-1,j-1)[c] * -1  + img.at<Vec3b>(i-1,j)[c] * 0   + img.at<Vec3b>(i-1,j+1)[c] * 1 + img.at<Vec3b>(i,j-1)[c] * -2 + img.at<Vec3b>(i,j)[c] * 0 + img.at<Vec3b>(i,j+1)[c] * 2  + img.at<Vec3b>(i+1,j-1)[c] * -1 + img.at<Vec3b>(i+1,j)[c] * 0   + img.at<Vec3b>(i+1,j+1)[c] * 1;
                value = value / 4;
                sobelx_16bit.at<Vec3s>(i,j)[c] = value;
            }
        }
    }
    
    Mat sobelx_8bit;
    convertScaleAbs(sobelx_16bit, sobelx_8bit);
    imshow("Sobel X 8 bit", sobelx_8bit);
    //imwrite("SobelX_8bit_q4.png", sobelx_8bit);
    //imwrite("SobelX_original.png", img);
    return 0;
}

int sobelY3x3(Mat &img, Mat &sobely_16bit ) {
    int value = 0;
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1 ; j++) {
            for (int c = 0; c<3; c++){
                value =  img.at<Vec3b>(i-1,j-1)[c] * 1  + img.at<Vec3b>(i-1,j)[c] * 2   + img.at<Vec3b>(i-1,j+1)[c] * 1 + img.at<Vec3b>(i,j-1)[c] * 0 + img.at<Vec3b>(i,j)[c] * 0 + img.at<Vec3b>(i,j+1)[c] * 0  + img.at<Vec3b>(i+1,j-1)[c] * -1 + img.at<Vec3b>(i+1,j)[c] * -2   + img.at<Vec3b>(i+1,j+1)[c] * -1;
                value = value / 4;
                sobely_16bit.at<Vec3s>(i,j)[c] = value;
            }
        }
    }
    Mat sobely_8bit;
    convertScaleAbs(sobely_16bit, sobely_8bit);
    //dst.convertTo(dst_8bit, CV_8UC3);
    imshow("Sobel Y", sobely_8bit);
    //imwrite("SobelY.png", sobely_8bit);
    //imwrite("Sobely_original.png", img);
    return 0;
}

int magnitude(Mat &sobelx_16bit, Mat &sobely_16bit, Mat &gradientmag_16bit ) {
    int value = 0;
    for (int i = 1; i < sobelx_16bit.rows - 1; i++) {
        for (int j = 1; j < sobelx_16bit.cols - 1 ; j++) {
            for (int c = 0; c<3; c++){
                value =  sqrt(sobelx_16bit.at<Vec3s>(i,j)[c] * sobelx_16bit.at<Vec3s>(i,j)[c] + sobely_16bit.at<Vec3s>(i,j)[c] * sobely_16bit.at<Vec3s>(i,j)[c]);
                //value = value / 4;
                gradientmag_16bit.at<Vec3s>(i,j)[c] = value;
            }
        }
    }
    Mat gradientmag_8bit;
    convertScaleAbs(gradientmag_16bit, gradientmag_8bit);
    //dst.convertTo(dst_8bit, CV_8UC3);
    imshow("Gradient Magtitude", gradientmag_8bit);
    //imwrite("GradientMagitude.png", gradientmag_8bit);
    //imwrite("GradientMagitude_original.png", img);
    return 0;
}

int blurQuantize(Mat &blur_16bit, Mat &quantize_16bit, int levels) {
    //blur_16bit.copyTo(quantize_8bit);
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
    Mat quantize_8bit;
    convertScaleAbs(quantize_16bit, quantize_8bit);
    imshow("blurQuantize", quantize_8bit);
    //imwrite("blurQuantize.png", quantize_8bit);
    return 0;
}


int cartoon(Mat &quantize_16bit, Mat &gradientmag_16bit, Mat &cartoon_16bit, int levels, int magThreshold ) {
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
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1 ; j++) {
            for (int c = 0; c<3; c++){
                contrast_16bit.at<Vec3s>(i,j)[c] = saturate_cast<uchar>( 1.2 * img.at<Vec3b>(i,j)[c] + 20 );
            }
        }
    }
    
    Mat contrast_8bit;
    convertScaleAbs(contrast_16bit, contrast_8bit);
    imshow("contrast_8bit", contrast_8bit);
    //imwrite("contrast_8bit.png", contrast_8bit);
    //imwrite("contrast_16bit_original.png", contrast_8bit);
    return 0;
}

int mirror(Mat &img, Mat &mirror_8bit ) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            for (int c = 0; c<3; c++){
                mirror_8bit.at<Vec3b>(i,j)[c] = img.at<Vec3b>(i,img.cols -j)[c];
            }
        }
    }
    //imwrite("mirror_8bit.png", mirror_8bit);
    //imwrite("mirror_8bit_original.png", mirror_8bit);
    return 0;
}

/*
int highfreq(Mat &img, Mat &blur_16bit, Mat &highfreq_16bit ) {
    for (int i = 0; i < blur_16bit.rows ; i++) {
        for (int j = 0; j < blur_16bit.cols  ; j++) {
            for (int c = 0; c<3; c++){
                highfreq_16bit.at<Vec3s>(i,j)[c] =  img.at<Vec3s>(i,j)[c] - blur_16bit.at<Vec3s>(i,j)[c];
            }
        }
    }
    Mat highfreq_8bit;
    convertScaleAbs(highfreq_16bit, highfreq_8bit);
    imshow("highfreq_8bit", highfreq_8bit);
    
    Mat overlay_16bit(img.rows, img.cols, CV_16SC3);
    overlay_16bit.create(img.rows, img.cols, CV_16SC3);
    
    Mat overlay_8bit;
    convertScaleAbs(overlay_16bit, overlay_8bit);
    imshow("overlay", overlay_8bit);
    
    
    return 0;
}
 */
