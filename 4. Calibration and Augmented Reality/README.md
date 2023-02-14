## Project 4: Calibration and Augmented Reality

To augment reality, this project calibrated extrinsic and intrinsic camera parameters, so a real world coordinates could be rigidlly transformed into pixel coordinates.  Now, having a coordinates of projected 3D objects onto 2D scene, a virtual object could then be created and synchronised with the movement of a target in a scene.

### Detect and Extract Chessboard Corners
The first task is to build a system for detecting a target and extracting target corners by OpenCV findChessboardCorners. A 9 x 6 checkerboard is used in this case, then used the OpenCV cornerSubPix in order to find a more precise coordinates up to subpixel.  

Below shows the numbers of corners found by findChessboardCorners and cornerSubPix, and also recorded the found first corner:

![image](https://user-images.githubusercontent.com/21034990/218807563-0c66802c-ecef-43fe-840a-2b5942228a77.png)

```
Corner size: 54
First corner: 558.579 387.984
```
### Select Calibration Images
In order to calibrate cameras from different view points, I have taken around 20 images with different translations, scales, and rotation of the checker board.  Then the 54 corners detected at each images, were saved in a pair with real world Euclidean coordinates.  Each corner in the checkerboard is 0.008m apart.

At this point, it's important to check if number of corners and Euclidean coordinates matches, as well as the sequence are aligned, because this is the step to feed in information to camera calibration to transform a 3D coordinates into a pixel.
 
### Calibrate the Camera
After saving 20 pairs of coordinates, I used cv::calibrateCamera to generate the calibration, and observed the RMS returned by the function.  Points to check at this step -
- The RMS is expected to fall between 0.1-1 pixel
- u0, v0 values should be close to image center (ie, image width/2, image height/2)
- Focal length at x and y should be close to each other

```
Error: 0.295002

camera matrix: 
[953.5073455891273, 0, 657.6923392325656;
 0, 953.5073455891273, 385.3010617447912;
 0, 0, 1]

distortion matrix 
[0.05480664702119955;
 -0.3857604992782505;
 0.007009501606443587;
 0.0006351146585209299;
 0.3822011478178676]
```

And write out the parameters into a xml file

<img src = "https://user-images.githubusercontent.com/21034990/180331456-ea225b58-383a-4b3a-8b16-b8e51bfbbeba.png" width = 400>

### Calculate Current Position of the Camera
Then used solvePNP to get the board's pose.

```
rvecs: [0.3105807656948578;
 -0.9458268253183723;
 -2.829166491667424]

tvecs: [0.1087392201451366;
 0.06097697801358722;
 0.203729751845701]
```

### Project Outside Corners or 3D Axes
Then project 3D real world coordinates onto 2D plane.  The 3D axes below shows the projection.

 ![image](https://user-images.githubusercontent.com/21034990/180331597-097432b3-a1d0-4242-9399-3948f5f88770.png)

### Create a Virtual Object
After real world corrdinates are successfully mapped, I can build a cube by real world dimensions.

 ![image](https://user-images.githubusercontent.com/21034990/180331637-d4a94e9f-6612-404f-8625-1855c4be8fdf.png)
 
 ![image](https://user-images.githubusercontent.com/21034990/180331645-c2e36ae7-a31d-4dc2-8bcb-1bd0bd51e43b.png)
  
### Detect Robust Features
Lastly, I have picked Harris corners to detect key points and showed the detection with the following patterns.
Harris Corners can detect corners/shape with irregular patterns.  

 ![image](https://user-images.githubusercontent.com/21034990/180331690-08b25355-6837-4c2e-802f-97f726e35006.png)
 ![image](https://user-images.githubusercontent.com/21034990/180331703-d097d9c9-85e9-4185-b062-d68b052f39af.png)
   
### Extension 1 and 2 - ArUco Markers + Insert virtual picture 
What did I plant during Spring break?  Place my sprout under the camera, then it grew up in a second into a green pepper!
 
This is not a straightforward task just to put a picture onto the checkers;  It takes also homography, image wrapping to move the virtual picture so it synchronise with movement.  How to get accurate corners to prospect the virtual picture took some challenges, since the aruco detectmarkers outputs in random sequences (though still in aligned pairs of markerId), accompany with endless parameters type transform for same meanings of parameters to accommodate similar-in-nature opencv function (vec<point2f>, vec<vector<point2f>>, vec<point2f>variable(4), vec<point> for the same single variable.)
 
https://youtu.be/0TRiQjddFG4
 
 
### Extension 3 : Detect multiple targets 
  ![image](https://user-images.githubusercontent.com/21034990/180387179-ef1d07ed-6b12-4ea6-8dc5-6785741b3290.png)

