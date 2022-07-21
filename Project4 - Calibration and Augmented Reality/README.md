## Project 4: Calibration and Augmented Reality

To augment reality, the project is to correlate extrinsic and intrinsic camera parameters, so a real world coordinates could be transformed into pixel coordinates.  Now, having a coordinates of projected 3D objects onto 2D scene, a virtual object could then be built and is able to synchronise movement of target in real world.

### Detect and Extract Chessboard Corners
Number of corners found by findChessboardCorners for a 9 x 6 checkerboard, then found precise coordinates with cornersubpix.  Below shows the numbers of corners found and also the first corner:

```
corner size: 54
558.579 387.984
```
###Select Calibration Images
Created a list for real world coordinates of 54 corners with real world dimension 0.008m apart.  Checked if number of points matches, as well as sequence of points if both lists are aligned. 
Each time when a user presses "s", new set of real world and findChessboardCorners would be appended and the current video frame would be saved.
 
###Calibrate the Camera
After saving 20 images, I pressed "c" to calibrate the camera.  
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
 
Calculate Current Position of the Camera
Then used solvePNP to get the board's pose.
rvecs: [0.3105807656948578;
 -0.9458268253183723;
 -2.829166491667424]

tvecs: [0.1087392201451366;
 0.06097697801358722;
 0.203729751845701]
 
Project Outside Corners or 3D Axes
Then project 3D real world coordinates onto 2D plane.  The 3D axes below shows the projection.
 
Create a Virtual Object
After real world corrdinates are successfully mapped, I can build a cube by real world dimensions.
   
Detect Robust Features
Lastly, I have picked Harris corners to detect key points and showed the detection with the following patterns.
Harris Corners can detect corners/shape with irregular patterns.  We might be able to use those feature points for custom made checkers or even checker-less for putting augmented reality into the image.
   
Extension 1 and 2 - ArUco Markers + Insert virtual picture 
What did I plant during Spring break?  Place my sprout under the camera, then it grew up in a second into a green pepper!
At first i tried projecting a video sequence.  Unfortunately the processing was so slow, even showing a single picture.  I did spend an hour to turn an animated gif into working video sequence though  
This is not a straightforward task just to put a picture onto the checkers;  It takes also homography, image wrapping to move the virtual picture so it synchronise with movement.  How to get accurate corners to prospect the virtual picture took some challenges, since the aruco detectmarkers outputs in random sequences (though still in aligned pairs of markerId), accompany with endless parameters type transform for same meanings of parameters to accommodate similar-in-nature opencv function (vec<point2f>, vec<vector<point2f>>, vec<point2f>variable(4), vec<point> for the same single variable.)
I also tried to carefully crop the sprout image so there's no white border.  However, it affected marker detection, so it is lesson to learn.
https://youtu.be/0TRiQjddFG4
 
 
Extension 3 : Detect multiple targets
My original goal was to build a simple shape over fixed markerId, but due to time constraint, i put fixed color of axis instead.  
 
Reflection
I hope opencv documentation could be more transparent about input parameter type.  Too much time spent on datatype conversion unfortunately.
Acknowledgement
Thanks Professor for helping out!
Good readings of Detection of ArUco Markers  https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
Virtual picture ideas come from https://learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/

![image](https://user-images.githubusercontent.com/21034990/180121722-b9cc168d-133e-4a20-8309-2bbf312e8d36.png)
