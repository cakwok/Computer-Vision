### Description
In this project, we are trying to recognise 2D objects by traditional computer vision approaches using morthological operations.  

Unlike human vision, of which we can differentiate rotated / scaled / transitioned objects instantly, in computer's world, it relies on statistics models, such as Moments, aspect ratio, percentage of pixels, unsupervisioned machine learning, to determine if an object could be identified and if not, how shall the object be learnt.

We will start the work by collecting statistics from 10 known objects.  Then, we are able to compare a unknown object with the database, and determine the object by different distance matrix - Euclidean distance to the closest neighbour, and Euclidean distance by KNN.

#### 1. Threshold input video
By streaming live output from iphone as video input, with objects placed in a white background, foreground is determined by keeping pixel intensity if they are > 100.  Ambient lighting affected my result.  In day time, the same thresholding could be achieved by intensity > 180.  At this step, background is removed, and foreground is kept as white to create binary images.

![image](https://user-images.githubusercontent.com/21034990/218847967-65621304-86e2-46de-b57a-306ef83b4d0e.png)
![image](https://user-images.githubusercontent.com/21034990/218848004-62eac2ec-0887-40ce-a326-0581a2d95807.png)

#### 2.  Clean up binary images
Then by glowing, pepper noise are removed.  It helps to reduce identified regionID by pepper noise, so results less false positive connected regions.<br>
![image](https://user-images.githubusercontent.com/21034990/218848069-7fd144f2-a5a0-4f47-9c79-65bfc37455bd.png)

#### 3.  Segment the image into regions
By OpenCV connectedComponentsWithStats with 8 way connectivity, the function uses connected segment algorithms to identify an object, namely regions, in computer vision.

An identified region is given with a region ID, centriods, area.  The capture below shows the largest regionaID identified and colored correspondingly.  

![image](https://user-images.githubusercontent.com/21034990/218848269-d69aa825-914e-4a8b-a668-38b2960d3c2e.png)

#### 4.  Compute features for each major region
After a region is identified, now we can compute features of the object. Raw moments m00, central moment nu02, nu20, nu11, aspect ratio, axis of least central moment were derived.  These features were chosen, because they are translational, scaling, rotational invariant.

![image](https://user-images.githubusercontent.com/21034990/218848329-c4f7ad3d-d75a-4351-b8af-d547d18c60b3.png)
![image](https://user-images.githubusercontent.com/21034990/218848459-b8b2a69d-a5b4-4d7b-9303-2623bbb08c0e.png)

#### 5.  Collect training data
By pressing "t" at the live video stream, user could attach a label onto an object and save into a csv based database.  Details would be shown at video link provided.


#### 6.  Classify new images
When an object appear in the video stream, the mentioned properties (Moments, aspect ratio, etc, ) would be computed a Euclidean distance against known objects in the database.  As shown in the capture below, the 3 new objects were computed, and attached a label with the closest neighbour with shortest Euclidean distance.

![image](https://user-images.githubusercontent.com/21034990/218848507-823ef8e4-faac-432e-82dc-7b7623f75d26.png)

#### 7.  Implement a different classifier
In order to improve performance of closet neighbour, at this step, KNN with closet 2 nearest neighbours was implemented. 

I observed not big difference when compared the result with the nearest neighbor, probably due to database and neighbor size is not large enough.

![image](https://user-images.githubusercontent.com/21034990/218848608-ed2a3152-8ebf-410a-8747-300c62d35f5d.png)

#### 8.  Evaluate the performance of your system
Below confusion matrix shows the true label verus classified label by closet neighbour.

![image](https://user-images.githubusercontent.com/21034990/218848790-a93b5494-d06a-44dd-925b-18bff9e88f25.png)

#### 9.  Capture a demo of your system working
https://youtu.be/neEPlZ1NsUE

https://youtu.be/KRhL2Q_Ddkk

Extension
1.  OpenCV Adaptive thresholding.
During the project set up, thresholding by a constant has imposed me challenges.  The first take of binary image was nice at day time, but at night time, I have to scale down threshold from 180 to 100.  Therefore, I was trying to study if there is a OpenCV function for thresholding, it would be useful even during day time, as shadow casted sometimes affect results.

However, having tried the adaptive threshold OpenCV function, it doesn't work like what I was looking for.  It averages all pixels as the output.



2.  Additional objects

I have added the below 5 objects into database, and recorded accuracy of the prediction.  Owing to limited objects i have on hand, the first 10 known objects in database are all real objects and took by live video.  The rest are downloaded from Internet.



left : object in database.  right : unknown object with predicted label.











3.  Better GUI

After obtaining an unknown object, user can press "e", look at the image, see the first time prediction, and attach a label.





Reflection
