//
//  Project2.cpp
//  CS 5330 Project 2 - Content-based Image Retrieval
//
//  Created by Casca on 7/2/2022.
//

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>       // For listing .jpg filenames codes
#include "csv_util.h"       // Professor's csv appending and reading
#include "Function.h"       // sobelx, sobely, gradient magnitude filters from project 1

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

bool sortcol(const pair<string, float> &col1, const pair<string, float> &col2) {        //sorting
  return (col1.second < col2.second);
}

int get_targeted_image_index(vector<char *> &filenames_csv, int &targeted_image_index, string &target_image) {
    auto it = find(filenames_csv.begin(), filenames_csv.end(), target_image);
    if (it != filenames_csv.end()) {
        targeted_image_index = it - filenames_csv.begin();
        cout << targeted_image_index << "\n";
    }
    return 0;
}

int main (){
    
    Mat img;
    const char *filename = "baseline.csv";      //------------------------- for writing csv
    const char *image_filename;                 //image files name in the folder "olympics"
    vector<float> image_data;                   //store pixel values into excel format
    int reset_file = 0;
    
    vector<char *> filenames_csv;               //for reading csv - first col of csv for filenames
    vector<vector<float>> data_csv;             //for reading csv - all col of csv except first for pixel values
    int echo_file = 0;
    
    string objectInDirectory;                  //path of files in "olympics"
    string target_image;                       //input image name by user to find top N matching images
    vector<float> target_image_feature_vector;
    
    // ------------------------------------
    // Question 1, Baseline matching
    // use 9 x 9 square in the middle of image as feature vector
    //
    // extract 9 x 9 square in the middle of image as feature vector
    // then write the pixel values into csv (as 9 x 9 x 3 = 243 columns)
    //
    // Feature vector - 3D RGB Histogram
    // Distance Metrics - Histogram Intersection
    // ------------------------------------
    
    for (const auto & entry : fs::directory_iterator("olympus")) {
        objectInDirectory = entry.path().string();
        image_filename = objectInDirectory.c_str();
        img = imread(image_filename);
        BaselineMatching(img, image_data, 9, 9);
        append_image_data_csv(filename, image_filename, image_data, reset_file );
       }

    // read baseline.csv
    // with filenames as filenames_csv, pixel values in n x 9 x 9 x 3 arrays as data_csv
    // -----------------------------------
    read_image_data_csv(filename, filenames_csv, data_csv, echo_file);
    
    cout << "CSV total number of rows: " << data_csv.size() << "\n";
    cout << "CSV total number of columns of pixel values: " << data_csv[0].size() << "\n\n";
    
    cout << "Question 1 - Enter filename to find top N matching images (eg, \"olympus/pic.1016.jpg\"): ";
    cin >> target_image;
    img = imread(target_image);
    
    // ------------------------------------------------- extract feature vectors from the targeted image
    int targeted_image_index;
    get_targeted_image_index(filenames_csv, targeted_image_index, target_image);
    
    int sum_of_square = 0;
    float sum_of_square_normalized = 0.0;
    vector<pair<string, float>> Question1_L2distance_result_table;
    
    // ------------------------------------------------- Calculate sum of square of difference
    
    for (int i = 0; i < data_csv.size(); i++) {
        
        sum_of_square = 0;
        
        for (int j = 0; j < data_csv[i].size(); j++) {
            sum_of_square += (data_csv[targeted_image_index][j] - data_csv[i][j]) * (data_csv[targeted_image_index][j] - data_csv[i][j]);
        }
        
        sum_of_square_normalized = sum_of_square / data_csv[0].size();  //normalize total pixel values = 9 x 9 x 3 = 243
        
        Question1_L2distance_result_table.push_back(make_pair(filenames_csv[i], sum_of_square_normalized));   //append result into a table
        
        //cout << "filenames_csv[i] " << filenames_csv[i] << "\n";
        //cout << "sum_of_square_normalized " << sum_of_square_normalized << "\n";
    }

    //cout << "Question1_L2distance_sorted_table \n";
    
    sort(Question1_L2distance_result_table.begin(), Question1_L2distance_result_table.end(), sortcol);      //sort to get the best of N result

    /*
    for (int i = 0; i < 10; i++ ) {
        cout << Question1_L2distance_result_table[i].first << "," << Question1_L2distance_result_table[i].second << "\n";
    }
     */
     
    cout << "Question 1 - toppest 10 matches \n";

    for (int k = 0; k < 10; k++) {
        cout << Question1_L2distance_result_table[k].first << "\n";
    }
    
    // -----------------------------------------------------------------------------------------------------
    // Question 2, Histogram matching
    // Feature Vector - 3D RGB Histogram for the upper part + 3D RGB Histogram for the lower part
    // Distance Metrics - Histogram intersection
    // -----------------------------------------------------------------------------------------------------
    
    cout << "\nQuestion 2 - Enter filename to find top N matching images (eg, \"olympus/pic.0164.jpg\"): ";
    cin >> target_image;
    img = imread(target_image);
    
    int dim[3] = {8, 8, 8};
    Mat histogramT = Mat::zeros(3, dim, CV_32SC3);
    int bucket = 8;
    int divisor = 256/bucket;
    int bix = 0; int gix = 0; int rix = 0;
 
    Mat histogram = Mat::zeros(3, dim, CV_32SC3);
    vector<pair<string, float>> Question2_L2distance_result_table;
    
    // -------------------------------------------------------- compute RGB 3D histogram of target image
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            bix = img.at<Vec3b>(i,j)[0] / divisor;
            gix = img.at<Vec3b>(i,j)[1] / divisor;
            rix = img.at<Vec3b>(i,j)[2] / divisor;
            histogramT.at<int>(rix, gix, bix) ++;
        }
    }
    
    for (const auto & entry : fs::directory_iterator("olympus")) {
        objectInDirectory = entry.path().string();
        image_filename = objectInDirectory.c_str();
        img = imread(image_filename);
        
        float sum_of_min = 0.0;
        int bix = 0; int gix = 0; int rix = 0;
        float value = 0.0; float value2 = 0.0;
        histogram = Mat::zeros(3, dim, CV_32S);
  
        // -------------------------------------------------------- compute RGB 3D histogram of all images
        for (int row = 0; row < img.rows; row++) {
            for (int col = 0; col < img.cols; col++) {
                bix = (img.at<Vec3b>(row,col)[0]) / divisor;
                gix = (img.at<Vec3b>(row,col)[1]) / divisor;
                rix = (img.at<Vec3b>(row,col)[2] ) / divisor;
                histogram.at<int>(rix, gix, bix) ++;
              }
        }
        
        // -------------------------------------------------------- compute histogram intersection
        for (int i = 0; i < bucket; i++) {
            for (int j = 0; j < bucket; j++) {
                for (int k = 0; k < bucket; k++) {
                    value = (float)histogramT.at<int>(i, j, k)/(img.rows * img.cols);
                    value2 = (float)histogram.at<int>(i, j, k) / (img.rows * img.cols);
                    sum_of_min = sum_of_min + min(value , value2);
                }
            }
        }
     
        //cout << "sum_of_min " << sum_of_min << "\n";
        //cout << image_filename << "," << sum_of_min << "\n";
        Question2_L2distance_result_table.push_back(make_pair(image_filename, sum_of_min));
    }
    //cout << "Question3_L2distance_sorted_table \n";

    sort(Question2_L2distance_result_table.begin(), Question2_L2distance_result_table.end(), sortcol);
    
    /*
    for (int i = 0; i < Question2_L2distance_result_table.size(); i++ ) {
        cout << Question2_L2distance_result_table[i].first << "," << Question2_L2distance_result_table[i].second << "\n";
    }
     */
     
    cout << "\nQuestion 2 - toppest 10 matches \n";

    for (int k = Question2_L2distance_result_table.size(); k > Question2_L2distance_result_table.size() - 10; k--) {
        cout << Question2_L2distance_result_table[k].first << "\n";
    }
    
    // -----------------------------------------------------------------------------------------------------
    // Question 3, Multi-Histogram matching
    // use 8 x 8 x 8 RGB 3D histogram bin as feature vector
    // but base on top half of the images and bottom half
    // -----------------------------------------------------------------------------------------------------
    cout << "\nQuestion 3 - Enter filename to find top N matching images (eg, \"olympus/pic.0274.jpg\"): ";
    cin >> target_image;
    img = imread(target_image);
    
    Mat histogramTtophalf = Mat::zeros(3, dim, CV_32SC3);
    Mat histogramTbottomhalf = Mat::zeros(3, dim, CV_32SC3);
    Mat histogramtophalf = Mat::zeros(3, dim, CV_32SC3);
    Mat histogrambottomhalf = Mat::zeros(3, dim, CV_32SC3);
    int b_upper = 0; int g_upper = 0; int r_upper = 0;
    int b_bottom = 0; int g_bottom = 0; int r_bottom = 0;
 
    vector<pair<string, float>> Question3_L2distance_result_table;
   
    // -------------------------------------------------------- compute multi RGB 3D histogram of target image
    for (int i = 0; i < img.rows; i++) {                                //compute 3d rgb histogram - top half
        for (int j = 0; j < img.cols; j++) {
            if (i < img.rows / 2 ) {
                b_upper = img.at<Vec3b>(i,j)[0] / divisor;
                g_upper = img.at<Vec3b>(i,j)[1] / divisor;
                r_upper = img.at<Vec3b>(i,j)[2] / divisor;
                histogramTtophalf.at<int>(r_upper, g_upper, b_upper) ++;
            }
            else {                                                      //compute 3d rgb histogram - bottom half
                b_bottom = img.at<Vec3b>(i,j)[0] / divisor;
                g_bottom = img.at<Vec3b>(i,j)[1] / divisor;
                r_bottom = img.at<Vec3b>(i,j)[2] / divisor;
                histogramTbottomhalf.at<int>(r_bottom, b_bottom, g_bottom) ++;
            }
        }
    }
 
    histogram = Mat::zeros(3, dim, CV_32SC3);
    float sum_of_min_upper = 0.0; float sum_of_min_bottom = 0.0;
    
    for (const auto & entry : fs::directory_iterator("olympus")) {
        objectInDirectory = entry.path().string();
        image_filename = objectInDirectory.c_str();
        img = imread(image_filename);
        
        float sum_of_min = 0.0;
        float value = 0.0; float value2 = 0.0; float value3 = 0.0; float value4 = 0.0;
        histogram = Mat::zeros(3, dim, CV_32S);
        b_upper = 0; g_upper = 0; r_upper = 0;
        b_bottom = 0; g_bottom = 0; r_bottom = 0;
        sum_of_min_upper = 0.0; sum_of_min_bottom = 0.0;
        histogramtophalf = Mat::zeros(3, dim, CV_32S);
        histogrambottomhalf = Mat::zeros(3, dim, CV_32S);
  
        // -------------------------------------------------------- compute multi RGB 3D histogram of all images
        for (int i = 0; i < img.rows; i++) {                             //Random images upper half - compute 3d rgb histogram
            for (int j = 0; j < img.cols; j++) {
                if (i < img.rows / 2 ) {
                    b_upper = img.at<Vec3b>(i,j)[0] / divisor;
                    g_upper = img.at<Vec3b>(i,j)[1] / divisor;
                    r_upper = img.at<Vec3b>(i,j)[2] / divisor;
                    histogramtophalf.at<int>(r_upper, g_upper, b_upper) ++;
                }
                else {                                                      //Random images lower half - compute 3d rgb histogram
                    b_bottom = img.at<Vec3b>(i,j)[0] / divisor;
                    g_bottom = img.at<Vec3b>(i,j)[1] / divisor;
                    r_bottom = img.at<Vec3b>(i,j)[2] / divisor;
                    histogrambottomhalf.at<int>(r_bottom, g_bottom, b_bottom) ++;
                }
            }
        }
        
        // -------------------------------------------------------- compute weighted histogram intersection, 0.5 * top + 0.5 * bottom
         for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 8; k++) {
                    value = (float)histogramtophalf.at<int>(i, j, k) / ((img.rows/2) * img.cols);
                    value2 = (float)histogramTtophalf.at<int>(i, j, k) / ((img.rows/2) * img.cols) ;
                    sum_of_min_upper = sum_of_min_upper + min(value , value2) ;
                    
                    value3 = (float)histogrambottomhalf.at<int>(i, j, k) / ((img.rows/2) * img.cols) ;
                    value4 = (float)histogramTbottomhalf.at<int>(i, j, k) / ((img.rows/2) * img.cols) ;
                    sum_of_min_bottom = sum_of_min_bottom + min(value3 , value4);
                }
            }
            sum_of_min = 0.5 * sum_of_min_upper + 0.5 * sum_of_min_bottom;
       }
        
        //cout << image_filename << "," << sum_of_min << "\n";
        Question3_L2distance_result_table.push_back(make_pair(image_filename, sum_of_min));
    }
    
    //cout << "Question3_L2distance_sorted_table \n";

    sort(Question3_L2distance_result_table.begin(), Question3_L2distance_result_table.end(), sortcol);
    
    /*
    for (int i = 0; i < Question3_L2distance_result_table.size(); i++ ) {
        cout << Question3_L2distance_result_table[i].first << "," << Question3_L2distance_result_table[i].second << "\n";
    }
     */
     
    cout << "Question 3 - toppest 10 matches \n";

    for (int k = Question3_L2distance_result_table.size(); k > Question3_L2distance_result_table.size() - 10; k--) {
        cout << Question3_L2distance_result_table[k].first << "\n";
    }

    // -----------------------------------------------------------------------------------------------------
    // Question 4, Texture and color matching
    // Feature Vector - 0.5 x RGB 3D Histogram + 0.5 x gradient magnitude histogram
    // Distance Metrics - Weighted average L1
    // -----------------------------------------------------------------------------------------------------

    cout << "\nQuestion 4 - Enter filename to find top N matching images (eg, \"olympus/pic.0535.jpg\") ";
    cin >> target_image;
    img = imread(target_image);
    
    Mat img_sobelx_16bit(img.rows, img.cols, CV_16SC3);
    Mat img_sobely_16bit(img.rows, img.cols, CV_16SC3);
    Mat img_gradientmag_16bit(img.rows, img.cols, CV_16SC3);
    img_sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
    img_sobely_16bit.create(img.rows, img.cols, CV_16SC3);
    img_gradientmag_16bit.create(img.rows, img.cols, CV_16SC3);
    
    Mat grey_imageT;
    cvtColor(img, grey_imageT, COLOR_BGR2GRAY);
    sobelX3x3(grey_imageT, img_sobelx_16bit);
    sobelY3x3(grey_imageT, img_sobely_16bit);
    magnitude(img_sobelx_16bit, img_sobely_16bit,  img_gradientmag_16bit );
    
    Mat img_gradientmag_8bit;
    convertScaleAbs(img_gradientmag_16bit, img_gradientmag_8bit);
    
    Mat histogramT_gradient;
    histogramT = Mat::zeros(3, dim, CV_32SC3);
    histogramT_gradient = Mat::zeros(3, dim, CV_32SC3);
    
    bix = 0; gix = 0; rix = 0;
    int b_gradient_img; int g_gradient_img = 0; int r_gradient_img = 0;
    
    vector<pair<string, float>> Question4_L2distance_result_table;
   
    for (int i = 0; i < img.rows; i++) {                                              //compute 3d rgb histogram
        for (int j = 0; j < img.cols; j++) {
            bix = img.at<Vec3b>(i,j)[0] / divisor;
            gix = img.at<Vec3b>(i,j)[1] / divisor;
            rix = img.at<Vec3b>(i,j)[2] / divisor;
            histogramT.at<int>(rix, gix, bix) ++;
            
            b_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[0] / divisor;         //compute 3d gradient magnitude histogram
            g_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[1] / divisor;
            r_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[2] / divisor;
            histogramT_gradient.at<int>(r_gradient_img, g_gradient_img, b_gradient_img) ++;
        }
    }
    
    histogram = Mat::zeros(3, dim, CV_32SC3);
    sum_of_min_upper = 0.0;  sum_of_min_bottom = 0.0;

    img_sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
    img_sobely_16bit.create(img.rows, img.cols, CV_16SC3);
    img_gradientmag_16bit.create(img.rows, img.cols, CV_16SC3);
    
    Mat histogram_gradient;
    Mat min_L2_distance_img;
    Mat grey_image;
    
    for (const auto & entry : fs::directory_iterator("olympus")) {      //for reading .jpg filenames
        objectInDirectory = entry.path().string();
        image_filename = objectInDirectory.c_str();
        img = imread(image_filename);
        
        float value = 0.0; float value2 = 0.0;
        bix = 0; gix = 0; rix = 0;
        int b_gradient_img = 0; int g_gradient_img = 0; int r_gradient_img = 0;

        histogram = Mat::zeros(3, dim, CV_32S);
        histogram_gradient = Mat::zeros(3, dim, CV_32SC3);
        img_sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
        img_sobely_16bit.create(img.rows, img.cols, CV_16SC3);
        img_gradientmag_16bit.create(img.rows, img.cols, CV_16SC3);

        sum_of_min_upper = 0.0; sum_of_min_bottom = 0.0;
  
        cvtColor(img, grey_image, COLOR_BGR2GRAY);
        sobelX3x3(grey_image, img_sobelx_16bit);
        sobelY3x3(grey_image, img_sobely_16bit);
        magnitude(img_sobelx_16bit, img_sobely_16bit,  img_gradientmag_16bit );
        convertScaleAbs(img_gradientmag_16bit, img_gradientmag_8bit);
        
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                bix = img.at<Vec3b>(i,j)[0] / divisor;
                gix = img.at<Vec3b>(i,j)[1] / divisor;
                rix = img.at<Vec3b>(i,j)[2] / divisor;
                histogram.at<int>(rix, gix, bix) ++;
                
                b_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[0] / divisor;
                g_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[1] / divisor;
                r_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[2] / divisor;
                histogram_gradient.at<int>(r_gradient_img, g_gradient_img, b_gradient_img) ++;
                
            }
        }
        
        value = 0.0; value2 = 0.0; float sum_of_difference = 0.0; float L2_distance = 0.0;
        
         for (int i = 0; i < 8; i++) {                                            //compute feature vector 1 - histogram target <> histogram random
            for (int j = 0; j < 8; j++) {                                         //compute feature vector 2 - gradient target <> gradient random
                for (int k = 0; k < 8; k++) {                                     //compute weighted L1/L2 distance
                    //Method 2 : color - texture, color - texture comparison
                    value = ((float)histogramT_gradient.at<int>(i, j, k) / (img.rows * img.cols)) * 0.5 + ((float)histogramT.at<int>(i, j, k) / ((img.rows) * img.cols)) * 0.5;
                    value2 = ((float)histogram_gradient.at<int>(i, j, k) / ((img.rows) * img.cols)) * 0.5 + ((float)histogram.at<int>(i, j, k) / ((img.rows) * img.cols)) * 0.5;
                    sum_of_difference += (value - value2) * (value - value2);
                    L2_distance += abs(value - value2);                                 //unremark if for L1 distance despite L2 named variable.
                    //L2_distance = sqrt(sum_of_difference);                            //unremark if for L2 distance!
                    
                }
            }
         }
  
        //cout << image_filename << "," << L2_distance << "\n";
        Question4_L2distance_result_table.push_back(make_pair(image_filename, L2_distance));
    }

    //cout << "Question4_L2distance_sorted_table \n";
    
    sort(Question4_L2distance_result_table.begin(), Question4_L2distance_result_table.end(), sortcol);
    /*
    for (int i = 0; i < Question4_L2distance_result_table.size(); i++ ) {
        cout << Question4_L2distance_result_table[i].first << "," << Question4_L2distance_result_table[i].second << "\n";
    }
     */
    cout << "\nQuestion 4 - toppest 10 matches \n";

    for (int k = 0; k < 10; k++) {
        min_L2_distance_img = imread(Question4_L2distance_result_table[k].first);
        cout << Question4_L2distance_result_table[k].first << "\n";
    }

    // -----------------------------------------------------------------------------------------------------
    // Question 5, Custom Design Part 1, use my own feature vector design
    // Feature Vector 1 - 0.5 * gradient magnitude histogram  + 0.5 RGB 3D histogram (both of the center 374 * 583 pixels of the target image)
    // Feature Vector 2 - 0.5 * gradient magnitude histogram  + 0.5 RGB 3D histogram (both of the center 374 * 583 pixels of the random image)
    //Distance Metrics - L1
    // -----------------------------------------------------------------------------------------------------

    cout << "\nEnter filename to find top N matching images (eg, \"olympus/pic.0562.jpg\"): ";
    cin >> target_image;
    img = imread(target_image);
    
    img_sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
    img_sobely_16bit.create(img.rows, img.cols, CV_16SC3);
    img_gradientmag_16bit.create(img.rows, img.cols, CV_16SC3);
    
    sobelX3x3(img, img_sobelx_16bit);
    sobelY3x3(img, img_sobely_16bit);
    magnitude(img_sobelx_16bit, img_sobely_16bit,  img_gradientmag_16bit );

    histogramT = Mat::zeros(3, dim, CV_32SC3);
    histogramT_gradient = Mat::zeros(3, dim, CV_32SC3);
    img_gradientmag_8bit = Mat::zeros(3, dim, CV_32SC3);
    
    bix = 0; gix = 0; rix = 0;
    b_gradient_img; g_gradient_img = 0; r_gradient_img = 0;
    int i = 0; int j = 0;
    
    for ( i = 177; i < 374; i++) {                                                      //compute histogram by taking center pixel
        for ( j = 29; j < 583; j++) {
            bix = img.at<Vec3b>(i,j)[0] / divisor;
            gix = img.at<Vec3b>(i,j)[1] / divisor;
            rix = img.at<Vec3b>(i,j)[2] / divisor;
            histogramT.at<int>(rix, gix, bix) ++;
            
            b_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[0] / divisor;         //compute histogram by taking center pixel of gradient magnitude
            g_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[1] / divisor;
            r_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[2] / divisor;
            histogramT_gradient.at<int>(r_gradient_img, g_gradient_img, b_gradient_img) ++;
        }
    }
    
    histogram = Mat::zeros(3, dim, CV_32SC3);
    sum_of_min_upper = 0.0;  sum_of_min_bottom = 0.0;
    
    img_sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
    img_sobely_16bit.create(img.rows, img.cols, CV_16SC3);
    img_gradientmag_16bit.create(img.rows, img.cols, CV_16SC3);
    
    vector<pair<string, float>> Question5_1_L2distance_result_table;
    
    cout << "\n";
    //cout << "Question 5 Part 1 \n";
    
    //for (const auto & entry : fs::directory_iterator("olympus\")) {                   //unremark if check 1000 dataset
    for (const auto & entry : fs::directory_iterator("olympusSubset")) {                //unremark if check 10 training dataset
        objectInDirectory = entry.path().string();
        image_filename = objectInDirectory.c_str();
        img = imread(image_filename);
        
        float sum_of_min = 0.0;
        float value = 0.0; float value2 = 0.0; float value3 = 0.0; float value4 = 0.0;
        bix = 0; gix = 0; rix = 0;
        int b_gradient_img = 0; int g_gradient_img = 0; int r_gradient_img = 0;

        histogram = Mat::zeros(3, dim, CV_32S);
        histogram_gradient = Mat::zeros(3, dim, CV_32SC3);
        img_gradientmag_8bit = Mat::zeros(3, dim, CV_32SC3);
        img_sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
        img_sobely_16bit.create(img.rows, img.cols, CV_16SC3);
        img_gradientmag_16bit.create(img.rows, img.cols, CV_16SC3);

        sum_of_min_upper = 0.0; sum_of_min_bottom = 0.0;

        sobelX3x3(img, img_sobelx_16bit);
        sobelY3x3(img, img_sobely_16bit);
        magnitude(img_sobelx_16bit, img_sobely_16bit,  img_gradientmag_16bit );
        convertScaleAbs(img_gradientmag_16bit, img_gradientmag_8bit);
                
        for (int i = 177; i < 374; i++) {
            for (int j = 29; j < 583; j++) {
                bix = img.at<Vec3b>(i,j)[0] / divisor;
                gix = img.at<Vec3b>(i,j)[1] / divisor;
                rix = img.at<Vec3b>(i,j)[2] / divisor;
                histogram.at<int>(rix, gix, bix) ++;
                
                b_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[0] / divisor;
                g_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[1] / divisor;
                r_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[2] / divisor;
                histogram_gradient.at<int>(r_gradient_img, g_gradient_img, b_gradient_img) ++;
            }
        }
        
        value = 0.0; value2 = 0.0; value3 = 0.0; value4 = 0.0;
        float sum_of_difference = 0.0; float L2_distance = 0.0;
        
         for (int i = 0; i < 8; i++) {                                                         //compute blended distance metrics;
            for (int j = 0; j < 8; j++) {                                                       //rgb * 0.5 + gradient * 0.5 as one feature vector
                for (int k = 0; k < 8; k++) {
                    value = ((float)histogramT_gradient.at<int>(i, j, k) / (374 * 583)) * 0.5 + ((float)histogramT.at<int>(i, j, k) / (374 * 583)) * 0.5;
                    value2 = ((float)histogram_gradient.at<int>(i, j, k) / (374 * 583)) * 0.5 + ((float)histogram.at<int>(i, j, k) / (374 * 583)) * 0.5;
                    sum_of_difference += (value - value2) * (value - value2);
                    //L2_distance = sqrt(sum_of_difference);                            //unremark if for L2 distance!
                    
                    L2_distance += abs(value - value2);                                 //unremark if for L1 distance despite L2 named variable.
                    
                }
            }
         }
        
        //cout << image_filename << "," << L2_distance << "\n";                          //unremark if verify sorting result
        Question5_1_L2distance_result_table.push_back(make_pair(image_filename, L2_distance));
    }

    sort(Question5_1_L2distance_result_table.begin(), Question5_1_L2distance_result_table.end(), sortcol);

    /*
    cout << "Question5_1_L2distance_sorted_table \n";

    for (i = 0; i < Question5_1_L2distance_result_table.size(); i++ ) {
        cout << Question5_1_L2distance_result_table[i].first << "," << Question5_1_L2distance_result_table[i].second << "\n";
    }
     */

    cout << "\nQuestion 5 part 1 - toppest 10 matches \n";

    for (int k = 0; k < 10; k++) {
        min_L2_distance_img = imread(Question5_1_L2distance_result_table[k].first);
        //imshow(Question5_1_L2distance_result_table[k].first, min_L2_distance_img);
        cout << Question5_1_L2distance_result_table[k].first << "\n";
    }
    
    //--------- Question 5 part 2

    cout << "\nEnter filename to find top N matching images (eg, \"olympus/pic.0746.jpg\"): ";
    cin >> target_image;
    img = imread(target_image);
    
    img_sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
    img_sobely_16bit.create(img.rows, img.cols, CV_16SC3);
    img_gradientmag_16bit.create(img.rows, img.cols, CV_16SC3);
    
    sobelX3x3(img, img_sobelx_16bit);
    sobelY3x3(img, img_sobely_16bit);
    magnitude(img_sobelx_16bit, img_sobely_16bit,  img_gradientmag_16bit );

    histogramT = Mat::zeros(3, dim, CV_32SC3);
    histogramT_gradient = Mat::zeros(3, dim, CV_32SC3);
    img_gradientmag_8bit = Mat::zeros(3, dim, CV_32SC3);
    
    bix = 0; gix = 0; rix = 0;
    b_gradient_img; g_gradient_img = 0; r_gradient_img = 0;
 
    for ( i = 0; i < img.rows; i++) {
        for ( j = 0; j < img.cols; j++) {
            bix = img.at<Vec3b>(i,j)[0] / divisor;
            gix = img.at<Vec3b>(i,j)[1] / divisor;
            rix = img.at<Vec3b>(i,j)[2] / divisor;
            histogramT.at<int>(rix, gix, bix) ++;
            
            b_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[0] / divisor;
            g_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[1] / divisor;
            r_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[2] / divisor;
            histogramT_gradient.at<int>(r_gradient_img, g_gradient_img, b_gradient_img) ++;
        }
    }
    
    histogram = Mat::zeros(3, dim, CV_32SC3);
    sum_of_min_upper = 0.0;  sum_of_min_bottom = 0.0;
    
    img_sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
    img_sobely_16bit.create(img.rows, img.cols, CV_16SC3);
    img_gradientmag_16bit.create(img.rows, img.cols, CV_16SC3);
    
    vector<pair<string, float>> Question5_2_L2distance_result_table;

    //for (const auto & entry : fs::directory_iterator("olympus")) {          //unremark for reading whole set
    for (const auto & entry : fs::directory_iterator("olympusSubset2")) {      //unremark for reading subset filenames
        objectInDirectory = entry.path().string();
        image_filename = objectInDirectory.c_str();
        img = imread(image_filename);
        
        float value = 0.0; float value2 = 0.0; float value3 = 0.0; float value4 = 0.0;
        bix = 0; gix = 0; rix = 0;
        int b_gradient_img = 0; int g_gradient_img = 0; int r_gradient_img = 0;

        histogram = Mat::zeros(3, dim, CV_32S);
        histogram_gradient = Mat::zeros(3, dim, CV_32SC3);
        img_gradientmag_8bit = Mat::zeros(3, dim, CV_32SC3);
        img_sobelx_16bit.create(img.rows, img.cols, CV_16SC3);
        img_sobely_16bit.create(img.rows, img.cols, CV_16SC3);
        img_gradientmag_16bit.create(img.rows, img.cols, CV_16SC3);

        sum_of_min_upper = 0.0; sum_of_min_bottom = 0.0;
  
        sobelX3x3(img, img_sobelx_16bit);
        sobelY3x3(img, img_sobely_16bit);
        magnitude(img_sobelx_16bit, img_sobely_16bit,  img_gradientmag_16bit );
        convertScaleAbs(img_gradientmag_16bit, img_gradientmag_8bit);
                
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                bix = img.at<Vec3b>(i,j)[0] / divisor;
                gix = img.at<Vec3b>(i,j)[1] / divisor;
                rix = img.at<Vec3b>(i,j)[2] / divisor;
                histogram.at<int>(rix, gix, bix) ++;
                
                b_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[0] / divisor;
                g_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[1] / divisor;
                r_gradient_img = img_gradientmag_8bit.at<Vec3b>(i,j)[2] / divisor;
                histogram_gradient.at<int>(r_gradient_img, g_gradient_img, b_gradient_img) ++;
            }
        }
        
        value = 0.0; value2 = 0.0; value3 = 0.0; value4 = 0.0;
        float sum_of_difference = 0.0; float L2_distance = 0.0;
        
         for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 8; k++) {
                    value = ((float)histogramT_gradient.at<int>(i, j, k) / (img.rows * img.cols)) * 0.5 + ((float)histogramT.at<int>(i, j, k) / (img.rows * img.cols)) * 0.5;
                    value2 = ((float)histogram_gradient.at<int>(i, j, k) / (img.rows * img.cols)) * 0.5 + ((float)histogram.at<int>(i, j, k) / (img.rows * img.cols)) * 0.5;
                    sum_of_difference += (value - value2) * (value - value2);
                    //L2_distance = sqrt(sum_of_difference);                            //unremark if for L2 distance!
                    L2_distance += abs(value - value2);                                 //unremark if for L1 distance despite L2 named variable.
                }
            }
         }
        //cout << image_filename << "," << L2_distance << "\n";                        //To verify path, L2 distance
        
        Question5_2_L2distance_result_table.push_back(make_pair(image_filename, L2_distance));
    }
    
    /*
    cout << "Question5_2_L2distance_result_table \n";
    
    sort(Question5_2_L2distance_result_table.begin(), Question5_2_L2distance_result_table.end(), sortcol);
    
    for (i = 0; i < Question5_2_L2distance_result_table.size(); i++ ) {
        cout << Question5_2_L2distance_result_table[i].first << "," << Question5_2_L2distance_result_table[i].second << "\n";
    }
    */
    cout << "Question 5 part 2 - toppest 10 matches \n";
    
    for (int k = 0; k < 10; k++) {
        min_L2_distance_img = imread(Question5_2_L2distance_result_table[k].first);
        //imshow(Question5_2_L2distance_result_table[k].first, min_L2_distance_img);
        cout << Question5_2_L2distance_result_table[k].first << "\n";
    }
    
    // -----------------------------------------------------------------------------------------------------
    // Question 6 Extension, Use opencv libriaries to compute
    // HSV histogram, correlation, chi square, histogram intersection
    // -----------------------------------------------------------------------------------------------------
    cout << "\nEnter filename to find top N matching images (eg, \"olympus/pic.0562.jpg\"): ";
    cin >> target_image;
    img = imread(target_image);
    
    Mat img_hsv;
    histogramT = Mat::zeros(3, dim, CV_32SC3);

    cvtColor(img, img_hsv, COLOR_BGR2HSV );
    
    cout << "Question 6 - Extension \n";
    cout << "file path,Correlation,Chi Square, Histogram Intersection \n";
    
    for (const auto & entry : fs::directory_iterator("olympusSubset")) {
        objectInDirectory = entry.path().string();
        image_filename = objectInDirectory.c_str();
        img = imread(image_filename);
        histogram = Mat::zeros(3, dim, CV_32SC3);
        
        cvtColor(img, img_hsv, COLOR_BGR2HSV );
    
        int h_bins = 8, s_bins = 8;
        int histSize[] = {h_bins, s_bins};

        float h_ranges[] = { 0, 180 };
        float s_ranges[] = { 0, 256 };

        const float* ranges[] = { h_ranges, s_ranges };
        
        int channels[] = { 0, 1 };
        calcHist(&img_hsv, 1 , channels, Mat(), histogramT, 2, histSize, ranges, true, false );
        normalize( histogramT, histogramT, 0, 1, NORM_MINMAX, -1, Mat() );
        
        calcHist(&img_hsv, 1 , channels, Mat(), histogram, 2, histSize, ranges, true, false );
        normalize( histogram, histogram, 0, 1, NORM_MINMAX, -1, Mat() );
        
        double correlation = compareHist( histogramT, histogram, 0 );
        cout << "img " << image_filename << "," << correlation << ",";
        
        double ChiSquare = compareHist( histogramT, histogram, 1 );
        cout <<  ChiSquare << ",";
        
        double HistogramIntersection = compareHist( histogramT, histogram, 2 );
        cout <<  HistogramIntersection << "\n";
        
        /*                                                                  //backup for calculating entropy of each image
        Mat logP;
        cv::log(histogram,logP);
        
        float img_entropy = -1*sum(histogramT.mul(logP)).val[0];
        float img_entropy = -1*sum(histogram.mul(logP)).val[0];
        
        cout << img_entropy << "\n";
         */
    }
    
    return 0;
}
