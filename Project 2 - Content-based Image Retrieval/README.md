## Description
In the project, we would be retrieving feature and texture from images, and return top N matching of a target image from 1106 random images.

Computers cannot see, therefore, spatial information, like the way how human vision recognise shapes, is not straight forward in computer's world.  Yet, an object's shape could be changing from time to time.

Therefore, in this project, on top of spatial information, we are also using statistical models as feature vectors to find the top N matches at our own choice. 

For the purpose of experimenting, I have applied sum of squared difference, L1 L2 distance, RGB histogram, HSV histogram, Chi-square, histogram intersection as feature vectors.  Depending on the nature of target images, different distance metrics fits better under certain conditions, and finally be able to retrieve common features and texture from images. 

### Baseline Matching
Feature vector - 9x9 pixel value in the middle of an image 

Distance metrics - Sum of Squared Difference

Target image -

![image](https://user-images.githubusercontent.com/21034990/188551396-645f8b01-bdaf-4b09-8c47-0e7566559d82.png)

Top matches -

![image](https://user-images.githubusercontent.com/21034990/188551418-cc742e03-08c3-4418-a9f2-5d65bfe446bc.png)

### Histogram Matching
Feature vector - 3D RGB Histogram

Distance Metrics - Histogram Intersection

Target image -

![image](https://user-images.githubusercontent.com/21034990/188551724-c05b0df5-79d6-4f6e-828c-08443c078aba.png)

Top matches -

![image](https://user-images.githubusercontent.com/21034990/188551763-c65b912e-3e5d-465a-b261-16cdd8ea49be.png)
![image](https://user-images.githubusercontent.com/21034990/188552611-14dd17f3-2d9b-49c6-8477-4f39a5c65580.png)
![image](https://user-images.githubusercontent.com/21034990/188552626-8c4eeaa5-9bab-43ff-a359-86d9fec2dc94.png)


### Multi-Histogram Matching
Feature Vector - 3D RGB Histogram for the upper part + 3D RGB Histogram for the lower part 

Distance Metrics - Histogram intersection

Target Image -

![image](https://user-images.githubusercontent.com/21034990/188551806-9f72c94d-13cc-4ec3-8920-09945b1f5ce0.png)

Top matches -
![image](https://user-images.githubusercontent.com/21034990/188551837-24be3dcc-b33c-4a2b-993c-75393a4b578d.png)

### Texture and Color
#### Scenario 1 
Feature Vector - 0.5 x RGB 3D Histogram + 0.5 x gradient magnitude histogram

Distance Metrics - Weighted average L1

#### Scenario 2
Feature Vector 1 - RGB Histogram 

Feature Vector 2 - Gradient magnitude histogram

Distance Metrics - Weighted average L2, 0.5, 0.5

Target Image -

![image](https://user-images.githubusercontent.com/21034990/188551906-038b5a8c-37ed-42c6-a7ea-502bad088597.png)

Top matches, Scenario 1 - 

The result differs from task 2 and task 3; Texture of bricks could be matched with different color intensity across spatial variance.

![image](https://user-images.githubusercontent.com/21034990/188551964-ae2f3033-5499-48d0-9147-857e46d901c2.png)

Top matches, Scenario 2 -

![image](https://user-images.githubusercontent.com/21034990/188551998-7b5688cd-2d3b-4eab-b039-38f310f45781.png)

### Custom Design 
In this task, I have chosen 2 objects to match with -  a car, and a green bin.

Target Image 1 - Car with 10 training data + 10 random images
Feature Vector 1 - 0.5 * gradient magnitude histogram  + 0.5 RGB 3D histogram (both of the center 374 * 583 pixels of the target image)

Feature Vector 2 - 0.5 * gradient magnitude histogram  + 0.5 RGB 3D histogram (both of the center 374 * 583 pixels of the random image)

Distance Metrics - L1

Target Image -

![image](https://user-images.githubusercontent.com/21034990/188552104-022a9454-aaa0-4e89-8619-bd9482c57840.png)

Top matches -

The test is satisfactory with 70% matching rate.  

![image](https://user-images.githubusercontent.com/21034990/188552143-1d1c7189-9f6b-4a40-92e3-7e1a9d0478a4.png)

Below shows the full testing dataset for comparison, 10 training dataset, 10 random dataset -

![image](https://user-images.githubusercontent.com/21034990/188552167-902823f3-a999-497b-a259-06dea4186a31.png)

Top matches with whole dataset -

Yes it still need a better way, but the first 8 images show something in common - a rectangular shape in the middle.  pic.0113 has a framed shape which might be close with gradient magnitude of the target image; the pic.0403 buddy always shows up in my test, no matter how i changed weightings or distances, i believe it's because of his glasses which shaped like a wheel in gradient magnitude.

![image](https://user-images.githubusercontent.com/21034990/188552228-8d1efa7e-f2fc-4a4e-99f1-e5c079a7bde0.png)

#### Target Image 2 - Green Bin
In this comparison, i have chosen 10 training dataset with green bin, another 10 random dataset with green grassland or full picture of green plants.

Feature Vector 1 - 0.5 * gradient magnitude histogram  + 0.5 RGB 3D histogram (whole image)

Feature Vector 2 - 0.5 * gradient magnitude histogram  + 0.5 RGB 3D histogram (whole random image)

Distance Metrics - L1 distance

Target Image -

![image](https://user-images.githubusercontent.com/21034990/188552259-3772c22a-ac23-4b49-9090-6e5809acebcc.png)

Top matches -

The test is satisfactory with 80% matching rate.

![image](https://user-images.githubusercontent.com/21034990/188552280-8f029a6d-c124-4d72-b25d-37e974dcb0d7.png)

Below shows the full testing dataset for comparison, 10 training dataset, 10 random dataset -

![image](https://user-images.githubusercontent.com/21034990/188552301-e2a0f071-4510-471a-9948-4b0575535008.png)

Result of top matches with whole dataset -

Yes it still need a better way, this time I tried to compare with the whole image instead of center, so as to compare the effect of partial and whole image.

![image](https://user-images.githubusercontent.com/21034990/188552322-fa10ad4e-42dc-4fd5-b4e6-e508f4db806a.png)

### Extension 1 : Correlation
Below shows the result of matching by HSV Histogram, correlation, against 10 training dataset.  The training set looks good matching.

![image](https://user-images.githubusercontent.com/21034990/188552355-2ecd8c6e-56df-4d48-8860-980d9c21af7c.png)

Therefore, i compare it with the whole dataset. It matches the circles quite well, while i haven't put take any edge detection (gradient magnitude/canny) in the computation.

![image](https://user-images.githubusercontent.com/21034990/188552378-9674a5a5-aa4a-4ad4-b8ec-3b7596f34738.png)

### Extension 2, Chi Square
Below shows the result of matching by HSV histogram, Chi square by dataset.

![image](https://user-images.githubusercontent.com/21034990/188552403-7a72234c-5e58-4b11-9ccf-58a974cf30ff.png)

When matching with the whole dataset, it looks like wheel shape is matched

![image](https://user-images.githubusercontent.com/21034990/188552425-8acb1a35-15e0-4dba-8331-4ca09ba98b9f.png)

### Extension 3, HSV Histogram Intersection
Below shows the result of matching by HSV histogram, histogram intersection.

![image](https://user-images.githubusercontent.com/21034990/188552462-98b4bef7-eeb8-4034-87cf-eccc57264f76.png)

But the result of whole dataset doesn't match very well.

![image](https://user-images.githubusercontent.com/21034990/188552485-4bc26eed-c58c-404e-a3c3-6c06c3cdbaf8.png)


