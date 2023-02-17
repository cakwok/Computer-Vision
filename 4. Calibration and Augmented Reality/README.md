## Project 4: Calibration and Augmented Reality

To augment reality, this project calibrated extrinsic and intrinsic camera parameters and extract key points in a scene, so a real world coordinates could be rigidlly transformed into pixel coordinates.  Now, having a coordinates of projected 3D objects onto a 2D scene, a virtual object could then be created and synchronised with the movement of a target in a scene.

In this project, I have worked on detecting and extracting key points using chessboard corners, Harris corners and ArUCO markers.

### Detect and Extract Chessboard Corners
The first task is to build a system for detecting a target and extracting target corners by OpenCV findChessboardCorners. A 9 x 6 checkerboard is used in this case, then used the OpenCV cornerSubPix in order to find a more precise coordinates up to subpixel.  

Below shows the numbers of corners found by findChessboardCorners and cornerSubPix, and also recorded the found first corner:

![image](https://user-images.githubusercontent.com/21034990/218807563-0c66802c-ecef-43fe-840a-2b5942228a77.png)

```
Corner size: 54
First corner: 558.579 387.984
```
#### Select Calibration Images
In order to calibrate cameras from different view points, I have taken around 20 images with different translations, scales, and rotation of the checker board.  Then the 54 corners detected at each images, were saved in a pair with real world Euclidean coordinates.  Each corner in the checkerboard is 0.008m apart.

At this point, it's important to check if number of corners and Euclidean coordinates matches, as well as the sequence are aligned, because this is the step to feed in information to camera calibration to transform a 3D coordinates into a pixel.
 
#### Calibrate the Camera
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

Save the parameters into a xml file so we can just need to calibrate camera once for the system.

<img width="450" alt="image" src="https://user-images.githubusercontent.com/21034990/218815137-cd0d011d-3618-4948-8c6d-cfd771d90248.png">

#### Calculate Current Position of the Camera
At this step, use cv::solvePNP to get the checkerboard's pose.

```
rvecs: [0.3105807656948578;
 -0.9458268253183723;
 -2.829166491667424]

tvecs: [0.1087392201451366;
 0.06097697801358722;
 0.203729751845701]
```

#### Project Outside Corners or 3D Axes
Now using the pose estimation, use cv::projectPoints to project 3D real world coordinates onto the 2D image plane and verify this result by drawing lines along the projected axes.

![image](https://user-images.githubusercontent.com/21034990/218816759-71dd58df-95d4-4572-bd04-0e0767ac0100.png)

#### Create a Virtual Object
Now, we are able construct a shape in the scene accordingly.

![image](https://user-images.githubusercontent.com/21034990/218817304-80dde80a-1e47-4966-a474-aa462b2c8a9b.png)
![image](https://user-images.githubusercontent.com/21034990/218817370-17227a45-412c-4e0e-9081-baa7dc314fb6.png)
  
### Detect and Extract Harris Corners
Besides choosing to extract features by checkerboard, Harris corners could be another option to detect key points.  

![image](https://user-images.githubusercontent.com/21034990/218817743-33c80e92-c973-4e9c-8517-82ed16cb553d.png)
<img src="https://user-images.githubusercontent.com/21034990/218817795-06017179-8e29-4f0c-8112-25f192b85c35.png" width = "210"><br>

### Detect and Extract ArUCO Markers
In this work, I have tried to identify key points using ArUCO Markers, and wrapped a virtual picture of the scene enclosed by ArUCO Markers.

This task has to use cv::findHomography to map the key points between ArUCO markers and an targeted image, then wrap image perspective so when the real object move, the virtual picture moves synchronisely. 
 
<img width="800" alt="image" src="https://user-images.githubusercontent.com/21034990/218818953-65a880df-6564-455a-b1e9-0ddf870fb330.png">
 
#### ArUCO Markers : Detection of multiple targets 
As a future work, ArUCO Markers could be used to identify multiple targets for different effects of virtual objects.
![image](https://user-images.githubusercontent.com/21034990/218821897-a5f5cad5-1574-45ea-a672-156290c84f5a.png)

