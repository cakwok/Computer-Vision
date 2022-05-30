//
//  imgDisplay.cpp
//  CS5330 Project 1
//
//  Created by Wing Man, Kwok on 22/1/2022.
//
//Tasks 1 - Read an image from a file and display it

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main() {
    string path = "Resources/casca.jpg";    //Read an image file
    Mat img = imread(path);
    imshow("Image", img);                   //Display it in a window
    
    int key = waitKey(0);
    
    while (true) {                          //check for a keypress.
        if (key == 113) {                   //if q, then quit
            destroyAllWindows();
            break;
        }
        else {
            key = waitKey(0);
        }
    }
    destroyAllWindows();
    return 0;
}

