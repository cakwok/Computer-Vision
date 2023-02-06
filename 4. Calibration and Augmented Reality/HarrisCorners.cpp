//
//
//  HarrisCorners to detect corners in live video.
//
//  Created by Casca on 21/3/2022.
//

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>            //OpenCV library for camera calibration

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    
    Mat img, img_g, img_norm, img_norm_scaled;
    
    VideoCapture cap(0);                            //Camera input source from Mac.  cap(0) for epoccam
    cap.read(img);
    
    int key = waitKey(1);
    
    // --- Initiate video loops
    
    while (true) {
        
        if (key == 113) {                           //press "q" to quit
            destroyAllWindows();
            break;
        }
        
        Mat img_HarrisCorner = Mat::zeros(img.size(), CV_32FC1);
    
        cvtColor(img, img_g, COLOR_BGR2GRAY);
        cornerHarris(img_g, img_HarrisCorner, 12, 13, 0.04);      //find detected by Harris corners
        normalize(img_HarrisCorner, img_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
        convertScaleAbs(img_norm, img_norm_scaled);
        
        for( int i = 0; i < img_norm.rows ; i++ ) {
            for( int j = 0; j < img_norm.cols; j++ ) {
                if( (int) img_norm.at<float>(i,j) > 100 ) {     //Circle corners if threshold exceeds
                    circle( img, Point(j,i), 5,  Scalar(0), 2, 8, 0);
                }
            }
        }
        
        imshow( "corners", img  );
        
        key = waitKey(30);
        cap.read(img);
    }
    
    return 0;
}
