### Description
In this project, I have implemented traditional computer vision approaches starting with morthological operations to solve 2D objects recognition and scene segmentation problem.  

Unlike human vision, of which we can differentiate rotated / scaled / translational objects instantly, in computer's world, machine relies on statistical modelling of connected regions to estimate if an object could be identified.  If not, how shall the object be learnt.

I started the work by collecting statistics, namely moments, aspect ratio, percentage of pixels, from 10 different objects.  Then, we are able to compare a unknown object with this built database, and predict the unknown objects by looking up the shortest 1)Euclidean distance to the closest neighbour, 2)Euclidean distance by K nearest neighbors (KNN).

#### 1. Thresholding an image with an input video, creating binary images
By streaming live video output from iphone as image source input, with objects placed in a white background, the foreground pixels or regions is extracted by matching pixel intensity values if they are > 100.  

Ambient lighting affected the results.  In day time, the region extraction could be achieved by thresholding intensity with values > 180.  

![image](https://user-images.githubusercontent.com/21034990/218853198-49974bcb-b7c7-422a-a7e6-1e8f9fe1509c.png)

At this step, background information is removed, and detected foreground regions are kept in white pixels, resulting images remain in 2 colors, as known as binary images.

![image](https://user-images.githubusercontent.com/21034990/218847967-65621304-86e2-46de-b57a-306ef83b4d0e.png)
![image](https://user-images.githubusercontent.com/21034990/218848004-62eac2ec-0887-40ce-a326-0581a2d95807.png)

#### 2.  Clean up binary images with morphological filtering
Then I pre-processed the image by shrinking pixel by pixel, so white pixels with neighbors next to dark pixels are changed to dark, and this process is commonly used to remove pepper noise in images.  This step helps to reduce number of false positive of connected regions(regionID).<br>
![image](https://user-images.githubusercontent.com/21034990/218848069-7fd144f2-a5a0-4f47-9c79-65bfc37455bd.png)

#### 3.  Segment the image into regions
By using the cv::connectedComponentsWithStats function with 8 way connectivity, of which the function implements connected segment algorithms, connected regions are identified with a regionID, corresponding centriods, area of the region, and a color index to visualize the regions.  Since false positives may not be completely eliminated, this step takes the top 10 largest regionID from the output.

![image](https://user-images.githubusercontent.com/21034990/218848269-d69aa825-914e-4a8b-a668-38b2960d3c2e.png)

#### 4.  Compute features for each major region
After a region is identified, now we can compute and extract features from the region. 

Features extracted in this work were raw moments m00, central moment nu02, nu20, nu11, aspect ratio, axis of least central moment.  These features were chosen because they are translational, scaling, rotational invariant in nature.

![image](https://user-images.githubusercontent.com/21034990/218848329-c4f7ad3d-d75a-4351-b8af-d547d18c60b3.png)
![image](https://user-images.githubusercontent.com/21034990/218848459-b8b2a69d-a5b4-4d7b-9303-2623bbb08c0e.png)

#### 5.  Collect training data
The features of the above would have to be collected and stored in a database.  At my system, by pressing "t" at the live video stream, a user could attach a label onto an object and save the trainings into a csv based database.  

#### 6.  Classify new objects
Now, when an unknown object appears in the video input, the system automatically collects the mentioned features(Moments, aspect ratio, etc).   These resulted feature vectors are then computed with the scaled Euclidean distance, and look up for the closest matching feature vector in the object database so as to identify the new object.

As shown in the capture below, I have provided another 3 new objects to the system, and the system is able to recognise the objects correctly, by comparing the features with the closest neighbour in the database..

![image](https://user-images.githubusercontent.com/21034990/218848507-823ef8e4-faac-432e-82dc-7b7623f75d26.png)

#### 7.  Implement a different classifier
To enhance prediction performance, at this step, instead of using Euclidean distance computation to look up for the closest neighbor, KNN with closet 2 nearest neighbours was implemented. Ideally it should be implemented with at least 3 neighbors.

![image](https://user-images.githubusercontent.com/21034990/218848608-ed2a3152-8ebf-410a-8747-300c62d35f5d.png)

#### 8.  Evaluate the performance of your system
So we understand the object recognition is estimated by distance matrix.  In order to know about the accuracy and performance of the system, I have built a confusion matrix to compare the count of true label verus predicted labels by nearest neighbor.

![image](https://user-images.githubusercontent.com/21034990/218848790-a93b5494-d06a-44dd-925b-18bff9e88f25.png)

#### 9. Objects from Internet
To understand if the system has been generalizing well, I have also downloaded online images to compare results.  In my system, if a user doesn't satisfy with the predicted label, the user can press "e" to attach a new label to create a new object in the database.

![image](https://user-images.githubusercontent.com/21034990/218866199-13c629c3-d5ad-49b5-ba2e-4edd34539c0d.png)
