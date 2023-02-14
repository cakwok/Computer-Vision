//
//  main.cpp
//  CS5330 - Project 3, Real-time Object 2-D Recognition
//
//  Created by Casca on 20/2/2022.
//
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>                     // For listing .jpg filenames codes
#include <fstream>
#include <math.h>
#include "csv_util.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int Threshold(Mat &img, Mat &img_threshold){        //Question 1 - Thresholding by taking white background away
    int averageRGB = 0;
    
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols - 1 ; j++) {
            averageRGB = (img.at<Vec3b>(i,j)[0] + img.at<Vec3b>(i,j)[1] + img.at<Vec3b>(i,j)[2]) / 3;
            if (averageRGB > 100 or averageRGB == 0) {
                img_threshold.at<Vec3b>(i,j)[0] = 0;
                img_threshold.at<Vec3b>(i,j)[1] = 0;
                img_threshold.at<Vec3b>(i,j)[2] = 0;
            }
            else {
                img_threshold.at<Vec3b>(i,j)[0] = 255;
                img_threshold.at<Vec3b>(i,j)[1] = 255;
                img_threshold.at<Vec3b>(i,j)[2] = 255;
            }
        }
    }
    return 0;
}

int CleanupBinary(Mat &img_threshold, Mat &img_BinaryCleanup) {     //Question  2 - Binary Cleanup

    img_threshold.copyTo(img_BinaryCleanup);
    
    for (int rows = 1; rows < img_threshold.rows - 1; rows++) {
        for (int cols = 1; cols < img_threshold.cols - 1 ; cols++) {
            for (int c = 0; c<3; c++){
                if (img_threshold.at<Vec3b>(rows+1,cols)[c] == 255) {
                    img_BinaryCleanup.at<Vec3b>(rows,cols)[c] = 255;
                }
                if (img_threshold.at<Vec3b>(rows,cols+1)[c] == 255 ) {
                    img_BinaryCleanup.at<Vec3b>(rows,cols)[c] = 255;
                }
                if (img_threshold.at<Vec3b>(rows,cols-1)[c] == 255 and img_threshold.at<Vec3b>(rows,cols)[c] == 0 ) {
                    img_BinaryCleanup.at<Vec3b>(rows,cols)[c] = 255;
                }
                if (img_threshold.at<Vec3b>(rows-1,cols)[c] == 255 and img_threshold.at<Vec3b>(rows,cols)[c] == 0 ) {
                    img_BinaryCleanup.at<Vec3b>(rows,cols)[c] = 255;
                }
            }
        }
    }
    return 0;
}

bool sortRegionArea(const pair<float, float> &col1, const pair<float, float> &col2) {        //sort for largest Region Area
  return (col1.second > col2.second);
}

bool sortSquareDistance(const pair<string, float> &col1, const pair<string, float> &col2) {  //sort for square distance
  return (col1.first < col2.first);
}

int main (){
    
    const char *filename = "training_data.csv";      //for reading csv
    int echo_file = 0;
    
    vector<char *> training_label;
    vector<vector<float>> data_csv;
    
    read_image_data_csv(filename, training_label, data_csv, echo_file);
    
    Mat img, img_threshold, img_BinaryCleanup;
    
    string objectInDirectory;
    string jpegfiles;
    
    Mat img_connectedcomponents;                                    //For Question 3 - connected components
    Mat stats(img.rows, img.cols, CV_32S);                          //For Question 3 - connected components
    Mat centroids(img.rows, img.cols, CV_64F);                      //For Question 3 - connected components
    int connectivity = 8;                                           //For Question 3 - Use 8 way connectivity

    VideoCapture cap(0);
    cap.read(img);
    
    int key = waitKey(1);
    
    vector<Vec3b> colors(10);                                       //Question3, label region with colors
    for(int label = 1; label < 10; ++label){
            colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    
    string training_data_label;
    float aspect_ratio = 0.0f;
    float SumofRoot;
    vector<float> UnknownFeatureVector;
    vector<pair<string, float>> square_distance;
    float min_d;
    string closet_neighbor;
    
    while (true) {
        
        imshow("Image1", img);
        
        closet_neighbor = "";
        
        if (key == 113) {       //press "q" to quit, Question 2
            destroyAllWindows();
            break;
        }
        
        if (key == 116) {       //press "t" to train, Extension, pre requsite for question 7
            Mat img_training; Mat img_training_threshold; Mat img_training_BinaryCleanup;
            
            for (const auto & entry : fs::directory_iterator("images")) {
                objectInDirectory = entry.path().string();
                jpegfiles = objectInDirectory.c_str();
                
                if (jpegfiles != "images/.DS_Store") {
                    img_training = imread(jpegfiles);
                    img_training.copyTo(img);
                }
            }
             
        }
        
        img.copyTo(img_threshold);
        Threshold(img, img_threshold);                              //-------- Question 1 ---- Threshold video
        
        img_threshold.copyTo(img_BinaryCleanup);
        CleanupBinary(img_threshold, img_BinaryCleanup);            //------- Question 2 ----- Clearn up binary image
        
        cvtColor(img_BinaryCleanup, img_BinaryCleanup, COLOR_BGR2GRAY); //------ Question 3 -- Segment images into regions
        int label_count = connectedComponentsWithStats(img_BinaryCleanup, img_connectedcomponents, stats, centroids, connectivity);
        
        //vector<Vec3b> colors(label_count);                          //-- Question 3, color regionID
        vector<pair<float, float>> RegionArea_table;                 //Create table with RegionID, Area
        RegionArea_table.clear();
        
        for(int label = 1; label < label_count; label++){           //Put label color into region
                RegionArea_table.push_back(make_pair(label, stats.at<int>(label, CC_STAT_AREA)));
        }
 
        sort(RegionArea_table.begin(), RegionArea_table.end(), sortRegionArea);     //Find largest region
        
        Mat img_labelled_region = Mat::zeros(img.rows, img.cols, CV_8UC3);
        vector<pair<float, float>> RegionalIDPixel0;
        vector<pair<float, float>> RegionalIDPixel1;
        
        RegionalIDPixel0.clear(); RegionalIDPixel1.clear();
        
        vector <Point> rotated_rectangle;
        rotated_rectangle.clear();
        Point vertex;
        
        for(int r = 0; r < img_connectedcomponents.rows; r++){                      //Color the largest region
            for(int c = 0; c < img_connectedcomponents.cols; c++){
                int label = img_connectedcomponents.at<int>(r, c);
                Vec3b &pixel = img_labelled_region.at<Vec3b>(r, c);
                
                if (stats.at<int>(label, CC_STAT_AREA) > 200 ) {                    //Question 3, ignore small regions
                    if (label == RegionArea_table[0].first) {
                        pixel = colors[label];
                        RegionalIDPixel0.push_back(make_pair(r,c));
                        vertex.x = c;
                        vertex.y = r;
                        rotated_rectangle.push_back(vertex);                    //Get pixel location to calculate Moments
                    }
                    else {
                        pixel = Vec3b(0,0,0);
                    }
                }
                else {
                    pixel = Vec3b(0,0,0);
                }
             }
         }
        
        //imshow("img_labelled_region", img_labelled_region);
        //waitKey(5000);
        
        Moments m;
        Point2f vtx[4];
        RotatedRect box = minAreaRect(rotated_rectangle);
        
        for (int i=0; i<1; i++){
            
            double centriodx = centroids.at<double>(RegionArea_table[i].first, 0);    //Question4, calculate centroid
            double centriody = centroids.at<double>(RegionArea_table[i].first, 1);
            
            m = moments(img_BinaryCleanup, false);                                     //Calculate Moments
            double alpha = (atan2(2 * m.mu11, (m.mu20 - m.mu02)))/2;
            alpha = alpha * M_PI / 180;
            
            Point pt;
            pt.x = centriodx;
            pt.y = centriody;
            
            box.points(vtx);
            
            for( i = 0; i < 4; i++ ) {                                              //Draw Oriented outbound rectangle
                line(img_labelled_region, vtx[i], vtx[(i+1)%4], Scalar(0, 255, 0), 1, LINE_AA);
            }
            
            float rot = 0.0f; float axis_length = 0.0f;
            
            aspect_ratio = min(box.size.height,box.size.width) / max(box.size.height, box.size.width);
            
            if (box.size.width < box.size.height) {
                rot = box.angle + 90;
                axis_length = box.size.height / 2;
            }
            else {
                rot = box.angle;
                axis_length = box.size.width / 2;
            }
            
            //Draw axis
            line(img_labelled_region, Point(pt.x, pt.y) - Point( axis_length * cos(rot * M_PI / 180),  axis_length * sin(rot * M_PI / 180)), Point(pt.x, pt.y) + Point(axis_length   * cos(rot * M_PI / 180), axis_length * sin(rot * M_PI / 180)), Scalar(0,0,255), 5);
            
            if (aspect_ratio != 0) {
                putText(img_labelled_region, "nu20: " +  to_string(m.nu20) + " aspect ratio: " + to_string(aspect_ratio), Point(5,100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
            }
            
            square_distance.clear();                                        //Question 6,classify new images
            square_distance.push_back(make_pair("0",0));
            UnknownFeatureVector.clear();
            
            UnknownFeatureVector.push_back(m.m00);                          //Calculate Eucliden Distance of new object
            UnknownFeatureVector.push_back(m.nu02);
            UnknownFeatureVector.push_back(m.nu20);
            UnknownFeatureVector.push_back(m.nu11);
            UnknownFeatureVector.push_back(min(box.size.height,box.size.width) / max(box.size.height, box.size.width));
            min_d = 1000000;
            closet_neighbor = "";
            
            int  unknown_max[5] = {0,0,0,0,0};
            int  unknown_min[5] = {0,0,0,0,0};
            
            for (int i = 1; i < 11; i++) {
                SumofRoot = 0.0f;
                 
                for (int j = 0; j < 5; j++) {    //Classify new image
                    SumofRoot += ((UnknownFeatureVector[j] - data_csv[i][j]) * (UnknownFeatureVector[j] - data_csv[i][j])) /(data_csv[11][j] * data_csv[11][j]);
                    
                }

                square_distance.push_back(make_pair(training_label[i],sqrt(SumofRoot)));
                
                //cout << training_label[i] << " " << square_distance[i].second << "\n";
               
                if (square_distance[i].second < min_d) {
                    closet_neighbor = training_label[i];
                    min_d = square_distance[i].second;
                }
                

            }
            
            //putText(img_labelled_region, "closet_neighbor: " + closet_neighbor, Point(25,200), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
            
            //waitKey(300000);
            
          }
        
        //------------- KNN start -------------                                 //Question 7, Find distance by KNN
        square_distance.push_back(make_pair("0", 0));
        for (int i = 12; i < 22; i++) {
            SumofRoot = 0.0f;
           
            for (int j = 0; j < 5; j++) {    //Classify new image
                SumofRoot += ((UnknownFeatureVector[j] - data_csv[i][j]) * (UnknownFeatureVector[j] - data_csv[i][j])) /(data_csv[11][j] * data_csv[11][j]);
            
            }
            
            square_distance.push_back(make_pair(training_label[i],sqrt(SumofRoot)));
            
          
        }
        sort(square_distance.begin(), square_distance.end(), sortSquareDistance);

        
        float knn_sum; knn_sum = 0.0f;
        float min_d_knn; min_d_knn = 10000000.0f;
        
        for (int i = 2; i < square_distance.size(); i+=2) {

            knn_sum = square_distance[i].second + square_distance[i+1].second;
             
            if (knn_sum < min_d_knn) {
                closet_neighbor = square_distance[i].first;
                min_d_knn = knn_sum;
            }
        }
        
      
        //if (key == 107 or key == 116) {
        if (key == 107) {
            putText(img_labelled_region, "closet_neighbor: " + closet_neighbor, Point(25,200), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
            putText(img_labelled_region, "2 Nearest Neighbor: " + closet_neighbor, Point(25,250), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
            imshow("Image", img_labelled_region);
            waitKey(300000);
        }
        
        
        //KNN stop -------------------------
        
        for (int i = 40; i < 45; i++) {                              //Extension, add additional objects
            SumofRoot = 0.0f;
             
            for (int j = 0; j < 5; j++) {    //Classify new image
                SumofRoot += ((UnknownFeatureVector[j] - data_csv[i][j]) * (UnknownFeatureVector[j] - data_csv[i][j])) /(data_csv[45][j] * data_csv[45][j]);
            }

            square_distance.push_back(make_pair(training_label[i],sqrt(SumofRoot)));
            
            if (sqrt(SumofRoot) < min_d) {
                closet_neighbor = training_label[i];
                min_d = sqrt(SumofRoot);
            }

        }
        
        if (aspect_ratio != 0) {
            putText(img_labelled_region, "closet_neighbor: " + closet_neighbor, Point(25,200), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        }
        
        imshow("Image", img_labelled_region);
        
        if (key == 116) {                   //press "t" to enter label of training dataset, Question 7
            waitKey(30000);
        }
        /*
        else {
            key = waitKey(20);
        }
         */
        
        if (key == 110 or key == 116) {       //press "n" to save training data, Question 5 collect training data
            cout << "Label: ";
            cin >> training_data_label;
            
            string csv_data;
            csv_data = training_data_label + "," +  to_string(m.m00) + "," + to_string(m.nu02) + "," + to_string(m.nu20)  + "," + to_string(m.nu11)  + "," + to_string(aspect_ratio) + "\n";

            ofstream outfile;
            outfile.open("training_data.csv", ios_base::app );
            outfile << csv_data ;
            outfile.close();
        }
        
        

        key = waitKey(20);
        
        if (key != 116) {
            cap.read(img);
        }
        
        if (key == 101) {       //press "e" for extension, learn opencv adaptive threholding
            Mat img_threshold_adaptive;
            Mat img_threshold_gray;
            Mat img_BinaryCleanup_adaptive;
            
            cvtColor(img, img_threshold_gray, COLOR_BGR2GRAY);
            adaptiveThreshold(img_threshold_gray, img_threshold, 180, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 12);
            
            imshow("img_threshold",  img_threshold);
            
            CleanupBinary(img_threshold, img_BinaryCleanup);
            
            imshow("img_BinaryCleanup",  img_BinaryCleanup);
            
            waitKey(30000);
        }
         
    }
    
    return 0;
    
}
