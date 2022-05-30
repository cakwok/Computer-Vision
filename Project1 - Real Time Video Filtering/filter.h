//
//  filter.h
//  vidDisplay.cpp
//
//  Created by Casca on 1/2/2022.
//

#ifndef FILTER_H
#define FILTER_H

using namespace cv;
using namespace std;

int greyscale( cv::Mat &img, cv::Mat &grey_image );
int blur5x5( cv::Mat &img, cv::Mat &blur_16bit );
int sobelX3x3(Mat &img, Mat &sobelx_16bit );
int sobelY3x3(Mat &img, Mat &sobelx_16bit );
int magnitude(Mat &sobelx_16bit, Mat &sobely_16bit, Mat &gradientmag_16bit );
int blurQuantize(Mat &blur_16bit, Mat &quantize_16bit, int levels);
int cartoon(Mat &quantize_16bit, Mat &gradientmag_16bit, Mat &cartoon_16bit, int levels, int magThreshold );
int contrast(Mat &img, Mat &contrast_16bit );
int mirror(Mat &img, Mat &mirror_8bit );
#endif /* FILTER_H */
