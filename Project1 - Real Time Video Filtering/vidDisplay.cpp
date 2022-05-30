//
//  vidDisplay.cpp version 1
//  CS5330 Project 1
//
//  Created by Wing Man, Kwok on 22/1/2022.
//
//
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "filter.h"

using namespace cv;
using namespace std;

int main(){
    
    Mat img, grey_image, dst;
    Mat blur_16bit(img.rows, img.cols, CV_16SC3);
    Mat blur_8bit;
    Mat sobelx_16bit(img.rows, img.cols, CV_16SC3);
    Mat sobelx_8bit;
    Mat sobely_16bit(img.rows, img.cols, CV_16SC3);
    Mat sobely_8bit;
    Mat gradientmag_16bit(img.rows, img.cols, CV_16SC3);
    Mat gradientmag_8bit;
    Mat quantize_16bit(img.rows, img.cols, CV_16SC3);
    Mat quantize_8bit;
    Mat cartoon_16bit(img.rows, img.cols, CV_16SC3);
    Mat cartoon_8bit;
    Mat contrast_16bit(img.rows, img.cols, CV_16SC3);
    Mat contrast_8bit;
    Mat mirror_8bit;
    string meme_text;
    
    int key = waitKey(1);
    int newkey = waitKey(1);
    
    VideoCapture cap(0);        //Task 2, open a video channel
    cap.read(img);
    
    while (true) {
        
        if (key == 113) {       //press "q" to quit, Question 2
            destroyAllWindows();
            break;
        }
        else if (key == 115) {  //press "s" to save, Question 2
            imwrite("Question2_output.png", img);
            destroyAllWindows();
        }
        else if (key == 103) {   //press "g" to greyscale by cvtcolor, Question 3
            cvtColor(img, grey_image, COLOR_BGR2GRAY);
            imshow("Image", grey_image);
        }
        else if (key == 104) {   //press "h" to greyscale by pixel access, Question 4
            greyscale(img, grey_image);
            imshow("Image", grey_image);
        }
        else if (key == 98) {   //press "b" to blur image by Guassian 5x5, Question 5
            blur5x5(img, blur_16bit);
            convertScaleAbs(blur_16bit, blur_8bit);
            imshow("Image", blur_8bit);
        }
        else if (key == 120) {   //press "x" for sobel x, Question 6
            sobelX3x3(img, sobelx_16bit);
            convertScaleAbs(sobelx_16bit, sobelx_8bit);
            imshow("Image", sobelx_8bit);
        }
        else if (key == 121) {   //press "y", for Sobel Y, Question 6
            sobelY3x3(img, sobely_16bit);
            convertScaleAbs(sobely_16bit, sobely_8bit);
            imshow("Image", sobely_8bit);
        }
        else if (key == 109) {   //press "m", Gradient Magnitude, Question 7
            sobelX3x3(img, sobelx_16bit);
            sobelY3x3(img, sobely_16bit);
            magnitude(sobelx_16bit, sobely_16bit, gradientmag_16bit);
            convertScaleAbs(gradientmag_16bit, gradientmag_8bit);
            imshow("Image", gradientmag_8bit);
        }
        else if (key == 108) {   //press "l", Quantizes an image, Question 8
            blur5x5(img, blur_16bit);
            blurQuantize(blur_16bit, quantize_16bit, 15);        //15 -> color level threshold, 255/15 per level
            convertScaleAbs(quantize_16bit, quantize_8bit);
            imshow("Image", quantize_8bit);
        }
        
        else if (key == 99) {   //press "c", Cartoonize an image, Question 9
            sobelX3x3(img, sobelx_16bit );
            sobelY3x3(img, sobely_16bit );
            magnitude(sobelx_16bit, sobely_16bit, gradientmag_16bit );
            blur5x5(img, blur_16bit);
            blurQuantize(blur_16bit, quantize_16bit, 15);
            cartoon(quantize_16bit, gradientmag_16bit, cartoon_16bit, 15, 20);  //20 -> gradient magnitude level.  if > 20, then draw black line

            convertScaleAbs(cartoon_16bit, cartoon_8bit);
            imshow("Image", cartoon_8bit);
        }
        else if (key == 111) {   //press "o", Contrast, Question 10
            contrast(img, contrast_16bit);
            convertScaleAbs(contrast_16bit, contrast_8bit);
            imshow("Image", contrast_8bit);
        }
        else if (key == 49) {   //press "1", Extensions 1, Mirror
            mirror(img, mirror_8bit) ;
            imshow("Image", mirror_8bit);
        }
        else if (key == 50) {   //press "2", Extension 2, Meme
            meme(img, meme_text);
            imshow("Image", img);
            key = -2;
        }
        else if (key == -2) {   //if user has pressed to enter meme in above, keep on showing meme
            putText(img, meme_text, Point(50, 150), FONT_HERSHEY_DUPLEX, 1.5, Scalar(255,192,203),2,false);
            imshow("Image", img);
        }
        else if (key == 51) {   //press "3", Video recording
            VideoWriter writer;
            int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
            double fps = 25.0;
            
            writer.open("video.avi", codec, fps, img.size(), CV_8UC3);
            
            while (true) {
                putText(img, "Recording...", Point(50, 150), FONT_HERSHEY_DUPLEX, 1.5, Scalar(0,0,255),2,false);
                writer.write(img);
                imshow("Image", img);
                cap.read(img);
                if (waitKey(5) >= 0)
                    break;
            }
        }
        else if (key == 52) {   //press "4", Blend 2 images
            blending(img, dst);
            imshow("Image", dst);
        }
        else if (key == -1 or key == 32) {            //key == null or key == space
            imshow("Image", img);
        }
        
        newkey = waitKey(1);
        
        if (newkey != -1) {
            key = newkey;
        }
        
        cap.read(img);        
    }
    return 0;
}
