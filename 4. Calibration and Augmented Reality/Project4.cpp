//
//  main.cpp
//  Project4.cpp
//
//  Created by Casca on 9/3/2022.
//
#include <iostream>
#include <opencv2/imgcodecs.hpp>          //OpenCV library for reading/writing, imread, imwrite
#include <opencv2/highgui.hpp>            //OpenCV library for user interfaces, display image, waitKey
#include <opencv2/imgproc.hpp>            //OpenCV library for processing images, threshold, drawing
#include <opencv2/calib3d.hpp>            //OpenCV library for camera calibration
#include <opencv2/aruco.hpp>              //OpenCV library for ArUco Checker
#include <filesystem>                     // For listing .jpg filenames codes
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::aruco;
namespace fs = std::filesystem;

int main(int argc, const char * argv[]) {
    
    Mat img; Mat img_g; Mat img_foundCorners;
    
    VideoCapture cap(0);                            //Camera input source from Mac.  cap(0) for epoccam
    cap.read(img);
    
    float RealWorldSquareDimension = 0.008f;      //0.8cm
    
    vector <Point2f> corners;                       //identified checkerboard pixel corners
    vector<vector<Point2f> > cornerlist;
    
    vector <Vec3f> RealWorldCoordinates;            //real world checkerboard coordinates
    vector<vector<Point3f> > RealWorldCoordinates_list;    

    vector <Point2f> lastcorners(corners);
    
    vector<Point3f> SquareRealWorldLength;
    
    bool found = false;
    
    Mat cameraMatrix = (Mat_<float>(3,3) << 1.0f, 0.0f, img.cols/2, 
                                            0.0f, 1.0f, img.rows/2, 
                                            0.0f, 0.0f, 1.0f);
    Mat distCoeffs;
    
    double rms = 0.0f;                              //initialise root mean square, defines errors of camera calibration
    
    Mat rvecs, tvecs;                               //initialise r vectors, t vectors
    
    int key = waitKey(1);
    
    /* ----------- Define real world coordinates
    Define real world coordinates by a 9 x 6 checkerboard.  
    Origin at leftmost corner, then another each corner is one unit of RealWorldSquareDimension away.
    In this case the checkerboard dimension is 0.8cm.
    The third dimension is always 0 at initialization, since this is referencing the plane of a checkerboard.
    */
    
    for (int r = 0; r < 6; r++ ){
        for (int c = 0; c < 9; c++) {
            SquareRealWorldLength.push_back(Point3f(c * RealWorldSquareDimension, r * RealWorldSquareDimension, 0.0f));
        }
    }
    // ------------------------
    
    //------- Read intrinic parameters ---
    FileStorage readxml("intrinsics.xml", cv::FileStorage::READ);
    
    readxml.open("intrinsics.xml", cv::FileStorage::READ);
    readxml["camera_matrix"] >> cameraMatrix;
    readxml["distortion_coefficients"] >> distCoeffs;
    
    cout << "intrinsic matrix:\n"; cout << cameraMatrix << "\n";
    cout << "distortion coefficients: \n"; cout << distCoeffs << endl;
    //readxml.release();
    // ------------------------
    
    //------- Prepare animated gif -----
    vector<Mat> RedPepperGif;
    Mat img_RedPepper;
    int RedPepperGif_counter = 0;
    
    VideoCapture vid_capture("RedPepper.gif");
    
    while (vid_capture.isOpened()) {
        vid_capture.read(img_RedPepper);

        if (img_RedPepper.empty()) {
            break;
        }
        else {
            RedPepperGif.push_back(img_RedPepper.clone());
        }
    }
    
    cout << "RedPepperGif size" << RedPepperGif.size() << "\n";
    RedPepperGif_counter = 0;

    // ------------------------
    
    //----------- Define Aruco Marker real world coordinates
    vector<Point3f> ArucoRealWorldLength(4);
    float ArucoWorldSquareDimension = 0.061f;      //Aruco Marker iphone size 6.1cm
    //float ArucoWorldSquareDimension = 0.05f;       //Aruco Marker print out size 5cm

    ArucoRealWorldLength[0] = Point3f(0.0f, 0.0f, 0.0f);
    ArucoRealWorldLength[1] = Point3f(0.061f, 0.0f, 0.0f);
    ArucoRealWorldLength[2] = Point3f(0.061f, 0.061f, 0.0f);
    ArucoRealWorldLength[3] = Point3f(0.0f, 0.061f, 0.0f);
    
    // ------------------------
    
    while (true) {
        
        imshow("Image1", img);
        
        if (key == 113) {       //press "q" to quit
            destroyAllWindows();
            break;
        }
        
        /* Question 1
        Build a system for detecting a target and extracting target corners.
        Draw the chessboard corners when it finds them
        */
        
        # findChessboardCorners is an OpenCV library to find the positions of internal corners of the chessboard.  Return True/False
        found = findChessboardCorners(img, Size(9,6), corners, found);  
        
        if (found) {
            img.copyTo(img_g);
            img.copyTo(img_foundCorners);
            cvtColor(img, img_g, COLOR_BGR2GRAY);
            
            /* Question 1, extract sub-pixel corrdinates using OpenCV cornerSubPix
               @params  corner                  Stores extracted corners
               @params  Size()                  Defines size of the neighborhood where it searches for corners.  Half side length of window, eg, (5*2+1) * (5*2+1)
               @params  Size(-1,-1)             not rejecting any neighborhood size
               @params  TermCriteria::EPS       difference between the current and previous function values less than threshold (0.001).
               @params  TermCriteria::COUNT:    The number of iterations should not exceed the specified maximum (40).
            */
            cornerSubPix(img_g, corners, Size(5,5), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40,0.001));
            
            /* draw chess board corners */
            drawChessboardCorners(img_foundCorners, Size(9,6), corners, found);
            
            /* print out how many corners it finds, along with the coordinates of the first corner */
            cout << "Corner size: " << corners.size() << "\n";       //Question 2, cout corners identified
            cout << "First corner: " << corners[0].x << " " <<  corners[0].y << "\n";    //Question 2, cout first corner
            
            imshow("Image1", img_foundCorners);
            
            /* Question 2, let user specifies a particular image should be used for the calibration  
               save the corner locations and the corresponding 3D world points.
            */
            
            //if user types 's', store the vector of corners found by findChessbordCorners into a corner_list.
            if (key == 115) {          
                cout << "corners list \n";
                for (int i = 0; i < corners.size(); i++ ) {
                    cout << corners[i].x << " " <<  corners[i].y << "\n";
                }
         
                cout << "\n";
                
                /* every time when user strokes "s", a list set of corners is appended to a list wrapping corners of each press of "s" 
                number of corners in the list has to be equal with real world coordinates */
                cornerlist.push_back(corners);
   
                /*  create a point_set that specifies the 3D position of the corners in world coordinates 
                number of corners in the list has to be equal with corner list*/
                RealWorldCoordinates_list.push_back(SquareRealWorldLength);
                
                /*
                int i = 0; int j=0;
                cout << "realworldcoordinates \n";
                for (int i = 0; i < RealWorldCoordinates_list.size(); i++ ) {
                    for (int j = 0; j < RealWorldCoordinates_list[i].size(); j++) {
                        cout << RealWorldCoordinates_list[i][j] << "\n";
                    }
                    cout << "i " << i << " j " << j << "\n";
                }
                */
            
                /* save images with identified corners */
                string checkerboardFile = "checkerboard_" + to_string(time(nullptr)) + ".png";
                imwrite(checkerboardFile, img_foundCorners);
            }
            
            //--- Question 4 Calculate Current Position of the Camera, grabs the locations of the corners, and then uses solvePNP to get the board's pose (rotation and translation). ------
            // solvePnP returns rotation, translation vectors that transform a 3D point to the camera coordinate frame
            
            solvePnP(SquareRealWorldLength, corners, cameraMatrix, distCoeffs, rvecs, tvecs);
            cout << "rvecs: \n" << rvecs << "\n";
            cout << "tvecs: \n" << tvecs << "\n";
            
            //---- Question 5 project Points and draw axis ----
            //use the projectPoints function to project the 3D points corresponding to the four outside corners of the chessboard onto the image plane in real time as the chessboard or camera moves around
            vector<Point2f> projected2dPoints;
            vector<Point3f> testing3dPoints;
            testing3dPoints.push_back(Point3f(0, 0, 0));
            testing3dPoints.push_back(Point3f(0.016, 0, 0));
            testing3dPoints.push_back(Point3f(0, 0.016, 0));
            testing3dPoints.push_back(Point3f(0, 0, -0.016));
            projectPoints(testing3dPoints, rvecs, tvecs, cameraMatrix, distCoeffs, projected2dPoints);
            //cout << "projected2dpoints \n";
            //cout << projected2dPoints << "\n";
            
            Mat DrawAxis;
            img.copyTo(DrawAxis);
            line(DrawAxis,projected2dPoints[0], projected2dPoints[1], Scalar(0,255,0), 2, LINE_4);
            line(DrawAxis,projected2dPoints[0], projected2dPoints[2], Scalar(255,0,0), 2, LINE_4);
            line(DrawAxis,projected2dPoints[0], projected2dPoints[3], Scalar(0,0,255), 2, LINE_4);
            imshow("projectPoints", DrawAxis);
        
            //--- Question 6 create a virtual object --
            vector <Point3f> virtual_box;
            virtual_box.push_back(Point3f(0, 0, 0));
            virtual_box.push_back(Point3f(0.016, 0, 0));
            virtual_box.push_back(Point3f(0, 0.016, 0));
            virtual_box.push_back(Point3f(0.016, 0.016, 0));
        
            virtual_box.push_back(Point3f(0, 0, -0.016));
            virtual_box.push_back(Point3f(0.016, 0, -0.016));
            virtual_box.push_back(Point3f(0, 0.016, -0.016));
            virtual_box.push_back(Point3f(0.016, 0.016, -0.016));
        
            projectPoints(virtual_box, rvecs, tvecs, cameraMatrix, distCoeffs, projected2dPoints);
        
            Mat DrawBox;
            img.copyTo(DrawBox);
        
            line(DrawBox,projected2dPoints[0], projected2dPoints[1], Scalar(0,0,255), 2, LINE_4);   //draw base square
            line(DrawBox,projected2dPoints[0], projected2dPoints[2], Scalar(0,0,255), 2, LINE_4);
            line(DrawBox,projected2dPoints[1], projected2dPoints[3], Scalar(0,0,255), 2, LINE_4);
            line(DrawBox,projected2dPoints[2], projected2dPoints[3], Scalar(0,0,255), 2, LINE_4);
        
            line(DrawBox,projected2dPoints[0], projected2dPoints[4], Scalar(0,0,255), 2, LINE_4);   //draw pillars
            line(DrawBox,projected2dPoints[1], projected2dPoints[5], Scalar(0,0,255), 2, LINE_4);
            line(DrawBox,projected2dPoints[2], projected2dPoints[6], Scalar(0,0,255), 2, LINE_4);
            line(DrawBox,projected2dPoints[3], projected2dPoints[7], Scalar(0,0,255), 2, LINE_4);
        
            line(DrawBox,projected2dPoints[4], projected2dPoints[5], Scalar(0,0,255), 2, LINE_4);   //draw top square
            line(DrawBox,projected2dPoints[4], projected2dPoints[6], Scalar(0,0,255), 2, LINE_4);
            line(DrawBox,projected2dPoints[5], projected2dPoints[7], Scalar(0,0,255), 2, LINE_4);
            line(DrawBox,projected2dPoints[6], projected2dPoints[7], Scalar(0,0,255), 2, LINE_4);
        
            imshow("Create Virtual box", DrawBox);
        }
        
        //--- Extension1&2 : ArUco Marker Detection and multiple object handling --
        vector<int> markerId;
        vector<vector<Point2f>> ArucoMarkerCorners, rejectedCorners;
        markerId.clear(); ArucoMarkerCorners.clear();
        Ptr parameters = DetectorParameters::create();
        Ptr dictionary = getPredefinedDictionary(DICT_4X4_50);
        
        Mat img_aruco;
        img.copyTo(img_aruco);
        
        detectMarkers(img, dictionary, ArucoMarkerCorners, markerId, parameters, rejectedCorners);
       
        if (ArucoMarkerCorners.size() == 2 and (markerId[0] != 2 and markerId[1] != 2 ) and (markerId[0] != 3 and markerId[1] != 3 )) {
            
            vector<Point2f> projected2dPoints;
            vector <Point3f> testing3dPoints;
            projected2dPoints.clear(); testing3dPoints.clear();
            
            testing3dPoints.push_back(Point3f(0, 0, 0));            //*real world coordinates different than checkerboard
            testing3dPoints.push_back(Point3f(0.061, 0, 0));        //Aruco marker is clockwise.  real world is horizontal
            testing3dPoints.push_back(Point3f(0, 0.061, 0));
            testing3dPoints.push_back(Point3f(0, 0, -0.061));
            
            Mat DrawAxis_Aruco;
            img.copyTo(DrawAxis_Aruco);
            
            for (int i = 0; i < markerId.size(); i++ ) {
                if (markerId[i] == 0) {                             //multiple object.  markerId 0 is marked all-blue axis
                    solvePnP(ArucoRealWorldLength, ArucoMarkerCorners[i], cameraMatrix, distCoeffs, rvecs, tvecs);

                    projectPoints(testing3dPoints, rvecs, tvecs, cameraMatrix, distCoeffs, projected2dPoints);
                    
                    line(DrawAxis_Aruco,projected2dPoints[0], projected2dPoints[1], Scalar(255,0,0), 2, LINE_4);
                    line(DrawAxis_Aruco,projected2dPoints[0], projected2dPoints[2], Scalar(255,0,0), 2, LINE_4);
                    line(DrawAxis_Aruco,projected2dPoints[0], projected2dPoints[3], Scalar(255,0,0), 2, LINE_4);
                 }
                
                else if (markerId[i] == 1) {                        //multiple object.  markerId 1 is marked all-green axis
                    solvePnP(ArucoRealWorldLength, ArucoMarkerCorners[i], cameraMatrix, distCoeffs, rvecs, tvecs);
                    
                    projectPoints(testing3dPoints, rvecs, tvecs, cameraMatrix, distCoeffs, projected2dPoints);
                    
                    line(DrawAxis_Aruco,projected2dPoints[0], projected2dPoints[1], Scalar(0,255,0), 2, LINE_4);
                    line(DrawAxis_Aruco,projected2dPoints[0], projected2dPoints[2], Scalar(0,255,0), 2, LINE_4);
                    line(DrawAxis_Aruco,projected2dPoints[0], projected2dPoints[3], Scalar(0,255,0), 2, LINE_4);
                }
            }
            
            drawDetectedMarkers(DrawAxis_Aruco,  ArucoMarkerCorners, markerId);     //Also draw markerId
            imshow("2 Aruco Markers", DrawAxis_Aruco);
        }
        
        //Extension 3, Put virtual objects onto markers
        if (ArucoMarkerCorners.size() == 4) {

            vector<Point2f> obj_corners(4);
            
            obj_corners[0] = Point2f(0, 0);                 //Prepare mapping of GreenPepper original coordinates
            obj_corners[1] = Point2f( img.cols, 0 );
            obj_corners[2] = Point2f( img.cols,  img.rows );
            obj_corners[3] = Point2f( 0,  img.rows );
        
            vector<Point2f> ArucoMarkerCorners_1d;
            ArucoMarkerCorners_1d.clear();
            
            for (int i = 0; i < ArucoMarkerCorners.size(); i++) {       //Prepare arucomarkers coordinates
                ArucoMarkerCorners_1d.push_back(ArucoMarkerCorners[i][0]);
            }
                        
            vector<Point2f> ArucoMarkerCorners_vertices(4);  //remeber to put dimension when vector is directly assigned values instead of!
            
            for (int i = 0; i < markerId.size(); i++ ) {    //Determine which corners to use for virtual picture frame
                if (markerId[i] == 0) {
                    ArucoMarkerCorners_vertices[0] = ArucoMarkerCorners[i][0];
                }
                else if (markerId[i] == 1) {
                    ArucoMarkerCorners_vertices[1] = ArucoMarkerCorners[i][1];
                }
                else if (markerId[i] == 2) {
                    ArucoMarkerCorners_vertices[2] = ArucoMarkerCorners[i][2];
                }
                else if (markerId[i] == 3) {
                    ArucoMarkerCorners_vertices[3] = ArucoMarkerCorners[i][3];
                }
            }
            
            vector<Point> point;            //Point type of ArucoMarkerCorners_vertices.  fillpoly accepts <Point> only
            point.push_back(ArucoMarkerCorners_vertices[0]);
            point.push_back(ArucoMarkerCorners_vertices[1]);
            point.push_back(ArucoMarkerCorners_vertices[2]);
            point.push_back(ArucoMarkerCorners_vertices[3]);
           
            Mat homography =  findHomography(obj_corners, ArucoMarkerCorners_vertices);  //Find homography between GreenPepper and marker coordinates
            Mat warpedImage;
            
            Mat filename;
            filename = imread("GreenPepper.jpeg");
            
            warpPerspective(filename, warpedImage,homography, img.size(), INTER_CUBIC);
            
            Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
            fillConvexPoly(mask, point, Scalar(255, 255, 255));
            
            Mat element = getStructuringElement( MORPH_RECT, Size(3,3) );   //Reduce edge effect of mask
            erode(mask, mask, element);
            
            warpedImage.copyTo(img_aruco, mask);
            
            imshow("warpedImage", img_aruco);
            
            /*                                                              //backup for animated gif as source
            if ( RedPepperGif_counter > RedPepperGif.size() - 1) {
                RedPepperGif_counter = 0;
            }
            else {
                RedPepperGif_counter++;
            }
             */
        }
        
        /* Question 3 calibrate camera and press w to write parameters into a file */
        /* press "c", Calibrate camera */
        if (key == 99) {          
            
            cameraMatrix = (Mat_<float>(3,3) << 1.0f, 0.0f, img.cols/2, 
                                                0.0f, 1.0f, img.rows/2, 
                                                0.0f, 0.0f, 1.0f);
            
            distCoeffs =  Mat::zeros(8, 1, CV_64F);
            
            float flag = CALIB_FIX_ASPECT_RATIO;
            
            //calibrateCamera function to generate the calibration and return RMS.  Returns rvecs, tvecs
            rms = calibrateCamera(RealWorldCoordinates_list, cornerlist, img.size(),  cameraMatrix, distCoeffs, rvecs, tvecs, flag);
                        
            cout << "Error: " << rms << "\n";
            cout << "camera matrix: " << "\n";
            cout << cameraMatrix << "\n";
            cout << "distortion matrix \n";
            cout << distCoeffs;
            
        }
        
        if (key == 119) {          //press "w"
            
            string csv_data;
            ofstream outfile;
            
            FileStorage fs("intrinsics.xml", cv::FileStorage::WRITE);
            fs << "camera_matrix" << cameraMatrix << "distortion_coefficients" << distCoeffs;
            fs.release();

        }
        
        key = waitKey(5);
        cap.read(img);
    }
    return 0;
}
