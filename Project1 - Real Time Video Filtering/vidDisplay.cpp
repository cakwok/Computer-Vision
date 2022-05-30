//
//  vidDisplay.cpp version 1
//  CS5330 Project 1
//
//  Created by Wing Man, Kwok on 22/1/2022.
//
//

#include <iostream>
#include </opt/homebrew/Cellar/opencv/4.5.4_3/include/opencv4/opencv2/core/core.hpp>
#include </opt/homebrew/Cellar/opencv/4.5.4_3/include/opencv4/opencv2/imgcodecs.hpp>
#include </opt/homebrew/Cellar/opencv/4.5.4_3/include/opencv4/opencv2/highgui.hpp>
#include </opt/homebrew/Cellar/opencv/4.5.4_3/include/opencv4/opencv2/imgproc.hpp>
#include </opt/homebrew/Cellar/opencv/4.5.4_3/include/opencv4/opencv2/opencv.hpp>
#include </Users/casca/Library/Mobile Documents/com~apple~CloudDocs/Northeastern Courses/Northeastern 2022 Spring CS5330/CS 5330 - HW1/filter.cpp>
 
/*
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
*/

using namespace cv;
using namespace std;

int main(){
    
    Mat img, grey_image, dst;
    Mat sobelx_16bit(img.rows, img.cols, CV_16SC3), sobely_16bit(img.rows, img.cols, CV_16SC3);
    Mat blur_16bit(img.rows, img.cols, CV_16SC3);
    Mat quantize_16bit(img.rows, img.cols, CV_16SC3);
    Mat highfreq_16bit(img.rows, img.cols, CV_16SC3);
    int key = waitKey(1);
    
    VideoCapture cap(0);        //Task 2, open a video channel
    cap.read(img);              //Task 2, create a window, and then loop, capturing a new frame, detect keystroke
    
    while (true) {
        
        imshow("Image", img);
        
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
            //imwrite("Question3_output_grey.png", grey_image);
            //imwrite("Question3_output_normal.png", img);
            imshow("cvtColor Gray", grey_image);
        }
        else if (key == 104) {   //press "h" to greyscale by pixel access, Question 4
            greyscale(img, grey_image);
            imshow("Grey", grey_image);
            //imwrite("Question4_rewrite_pixel.png", grey_image);
   
        }
        else if (key == 98) {   //press "b" to blur image by Guassian 5x5, Question 5
            blur_16bit.create(img.rows, img.cols, CV_16SC3);
            img.convertTo(img, CV_16SC3);
            blur5x5(img, blur_16bit);
            Mat blur_8bit;
            convertScaleAbs(blur_16bit, blur_8bit);
            imshow("blur", blur_8bit);
            //imwrite("Question5_blur.png", blur_8bit);
        }
        else if (key == 120) {   //press "x" for sobel x, Question 6
            sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
            sobelX3x3(img, sobelx_16bit );
            
        }
        else if (key == 121) {   //press "y", for Sobel Y, Question 6
            sobely_16bit.create(img.rows, img.cols, CV_16SC3);
            sobelY3x3(img, sobely_16bit );

        }
        else if (key == 109) {   //press "m", Gradient Magnitude, Question 7
    
            sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
            sobelX3x3(img, sobelx_16bit );
            
            sobely_16bit.create(img.rows, img.cols, CV_16SC3);
            sobelY3x3(img, sobely_16bit );
           
            Mat gradientmag_16bit(sobely_16bit.rows, sobely_16bit.cols, CV_16SC3);
            gradientmag_16bit.create(sobely_16bit.rows, sobely_16bit.cols, CV_16SC3);
            magnitude(sobelx_16bit, sobely_16bit, gradientmag_16bit );
        }
        else if (key == 108) {   //press "l", Quantizes an image, Question 8
            blur_16bit.create(img.rows, img.cols, CV_16SC3);
            img.convertTo(img, CV_16SC3);
            blur5x5(img, blur_16bit);
            
            quantize_16bit.create(blur_16bit.rows, blur_16bit.cols, CV_16SC3);
            blurQuantize(blur_16bit, quantize_16bit, 15);
        }
        
        else if (key == 99) {   //press "c", Cartoonize an image, Question 9
            
            sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
            sobelX3x3(img, sobelx_16bit );
            
            sobely_16bit.create(img.rows, img.cols, CV_16SC3);
            sobelY3x3(img, sobely_16bit );
           
            Mat gradientmag_16bit(sobely_16bit.rows, sobely_16bit.cols, CV_16SC3);
            gradientmag_16bit.create(sobely_16bit.rows, sobely_16bit.cols, CV_16SC3);
            magnitude(sobelx_16bit, sobely_16bit, gradientmag_16bit );
            
            blur_16bit.create(img.rows, img.cols, CV_16SC3);
            img.convertTo(img, CV_16SC3);
            blur5x5(img, blur_16bit);
            
            quantize_16bit.create(blur_16bit.rows, blur_16bit.cols, CV_16SC3);
            blurQuantize(blur_16bit, quantize_16bit, 15);
            
            Mat cartoon_16bit(quantize_16bit.rows, quantize_16bit.cols, CV_16SC3);
            cartoon_16bit.create(quantize_16bit.rows, quantize_16bit.cols, CV_16SC3);
            cartoon(quantize_16bit, gradientmag_16bit, cartoon_16bit, 15, 20);

            Mat cartoon_8bit;
            convertScaleAbs(cartoon_16bit, cartoon_8bit);
            imshow("cartoon", cartoon_8bit);
            //imwrite("cartoon.png", cartoon_8bit);
        }
        else if (key == 111) {   //press "o", Contrast, Question 10
            Mat contrast_16bit(img.rows, img.cols, CV_16SC3);
            contrast_16bit.create(img.rows,img.cols, CV_16SC3);
            contrast(img, contrast_16bit ) ;
            
        }
        else if (key == 49) {   //press "1", Extensions 1, Mirror
            Mat mirror_8bit;
            img.copyTo(mirror_8bit);
            mirror(img, mirror_8bit) ;
            imshow("mirror", mirror_8bit);
        }
        else if (key == 50) {   //press "2", Extension 2, Meme
            string text;
            cout << "What is meme of the day? \n";
            getline(cin, text);
            cout << text << "\n";
            putText(img, text, Point(50, 150), FONT_HERSHEY_DUPLEX, 1.5, Scalar(255,192,203),2,false);
            imshow("meme", img);
            imwrite("meme.png", img);
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
            /*
            blur_16bit.create(img.rows, img.cols, CV_16SC3);
            img.convertTo(img, CV_16SC3);
            blur5x5(img, blur_16bit);
            
            highfreq_16bit.create(img.rows, img.cols, CV_16SC3);
            highfreq(img, blur_16bit, highfreq_16bit);
         */
            double alpha = 0.5; double beta; 
            Mat cny;
            beta = ( 1.0 - alpha );
            cny = imread("cny.jpg");
            addWeighted( cny, alpha, img, beta, 0.0, dst);
            
            imshow("cny", dst);
        }
        
        key = waitKey(5);
        cap.read(img);  
        
    }
    return 0;
}
