# OCR Using CNN
In this project, I have built, trained, analyzed, modified a Convolutional Neural Network CNN to learn recognizing numbers and Greek letters, alongside with creating feature embeddings and transfer learning.

The objective of the project is to achieve recognition accuracy of over 90% for handwritten characters.

### Method

To prepare a real life testing dataset, as shown below in the capture, I pre-processed my own set of handwrittings into binary images.  This is to match with the format of the MNIST dataset. 

At right upper corner, the unoptimized prediction result is displayed.  As we can count from the "prediction" value, the model achieved a correct prediction rate of approximately 60% in the test set. This indicates the model requires additional tuning and optimization to achieve higher accuracy.

![image](https://user-images.githubusercontent.com/21034990/176381947-5a45a6b7-511a-4099-8e65-5be10de0ca08.png)

### Main codes 
Q1a - 1e    Q1a - CnnCoreStructure.py<br>
Q1f - 1g    Q1f - Read trained MNIST network3.py<br>
Q2          Q2a - Examine network.py<br>
Q3          Q3 Create a digit embedding.py<br>
Q4i         Q4i - Design your own experiment Epoch.py<br>
Q4ii        Q4ii - Design your own experiment batch size.py<br>
Q4iii       Q4iii - Design your own experiment batch normalization.py<br>

### EDA
The MNIST digit data consists of a training dataset of 60k 28x28 labeled digits and a testing dataset of 10k 28x28 labeled digits. The dataset was imported directly from the torchvision package as torchvision.datasets.MNIST.   Below capture shows the examples of the MNIST digit dataset with the corresponding ground truth.

<img src = "https://user-images.githubusercontent.com/21034990/177019922-2f674cf3-daf6-44cd-9e23-1e7fea3aa37c.png" width = 400>

### Built the CNN
Built the network with 2 convolution layers with max pooling and dropout, then trained the model for 5 epochs with batch size = 64.  

<img src = "https://user-images.githubusercontent.com/21034990/219908228-7e2e7bf2-e7d5-4e2c-82f9-88d94cec1c33.png" width = 400>

Upon completing 5 epochs, the plotted data indicates that the negative log likelihood loss function has shown a significant reduction in value, from over 2 to plateau around 0.5, after processing approximately 60,000 examples which is roughly equal to 1 epoch.

<img src = "https://user-images.githubusercontent.com/21034990/177019931-3d9b189b-c1cb-46df-9703-2f54866ce848.png" width = 400>

And by observing from the network logging, we can further visualize that the training has significant improvement at the second epoch.

```
Test set: Avg. loss: 2.3065, Accuracy: 1101/10000 (11%)

Test set: Avg. loss: 0.2158, Accuracy: 9337/10000 (93%)

Test set: Avg. loss: 0.1350, Accuracy: 9596/10000 (96%)

Test set: Avg. loss: 0.1036, Accuracy: 9655/10000 (97%)

Test set: Avg. loss: 0.0929, Accuracy: 9701/10000 (97%)

Test set: Avg. loss: 0.0786, Accuracy: 9754/10000 (98%)
```


After the trainings, we can see the system is able to classify all training samples correctly.

<img src = "https://user-images.githubusercontent.com/21034990/177020029-8ffe6900-d00a-4c49-afb5-23c699d0652e.png" width = 400>

Table shows the observed network output (log softmax), ground truth, argmax
<img src = "https://user-images.githubusercontent.com/21034990/219908252-47c2cabd-4d9e-478e-95cb-45f830e88246.png" width = 600>


To assess the system's ability to recognize real-life handwritten numbers, I used my own handwritten numbers as the testing dataset. The images were scaled down from 1000 pixels resolution to 28x28 pixels to match the format of the MNIST dataset. 

The system could only achieve 60% accuracy at this stage, but we will see later on it could be fixed.

<img src = "https://user-images.githubusercontent.com/21034990/177020056-a88ca893-9d17-4ab5-b7df-ad99cf590942.png" width = 400>

### Examine the network

To analzye how the first layer of the CNN processes data, I have accessed the weights by model.conv1.weight.  The resulting tensors [10, 1, 5,5] in shape, representing 10 filters witha patch size of 5x5.  eg, To access the ith 5x5 filter weights, I used weights[i,0] for the observation.

<img src = "https://user-images.githubusercontent.com/21034990/177020072-10b6dbf9-f0a9-496c-bba3-73c386a161a4.png" width = 400>

```
Parameter containing:

tensor([[[[ 0.0400,  0.3058,  0.0781,  0.1083, -0.0687],

          [ 0.2779,  0.3820,  0.0521,  0.3355,  0.1946],

          [ 0.3076,  0.1876,  0.2131,  0.0889,  0.3187],

          [ 0.0804, -0.0923,  0.0143, -0.0979,  0.0445],

          [-0.1237,  0.0518, -0.2613, -0.1508, -0.0363]]],
```

Then I used the fiter2D OpenCV function to apply the 10 filters to a training example, and examined the effect of the filters.  Through this analysis, it looks like the 5th filters (from left to right) could be a ridge detection as the boundaries of the filters are all large negatives values.

<img src = "https://user-images.githubusercontent.com/21034990/177020089-9992e96c-71d5-4d92-9b80-3900813899e2.png" width = 400>

Next, I built a truncated model using only the first 2 convolutional layers in order to observe the weight of the second layer.  This was accomplished by creating a subclass of the model, overriding the forward method, and displaying a few channels out of the 20 channels that are 4x4 in size.  

Observed from the output, the sixth image appears to be a sobel y filter.  This provides insight into what the second layer was trying to learn.

<img src = "https://user-images.githubusercontent.com/21034990/177020099-71c2df2a-5189-407a-ae87-1907d8aff43b.png" width = 400>

#### Create a digit embedding space

Leveraging the model, I built another submodel that terminates at the dense layer with 50 outputs and loaded the learnt weight from MNIST for transfer learning.  Then I applied a Greek letter dataset to obtain a set of 27 x 50 element vectors.  Instead of continuing with the softmax operations, the vectors were extracted and computed the sum squared distance between sample-wise vector and passed to a KNN classifier.

By using this technique, I can use my own choice of classifiers. (so now the network only has 2 convolutional layers with no output layers)

From the KNN classifier results, the system has reached 2/3 correctness, corresponding to a 67% accuracy.

<img src = "https://user-images.githubusercontent.com/21034990/177020106-0deaa1c8-377d-4a21-bfd6-a938903f4b23.png" width = 400>

#### Design my own experiment
To optimize prediction performance, I have experimented the follow ablation settings:
- different number of epochs(5, 10, 15, 20, 25, 30) 
- batch sizes 64, 128, 192, 256, 320, 384
- replaced 2 dropout layers with 2 batch normalisation layers

in total 6 x 6 x 2(with batchnormal or with dropout) 72 variations/combinations.  

The best result was obtained with 15 epochs and a batch size of 256.

After tuning, the system was able to correctly classify all digits.

<img src = "https://user-images.githubusercontent.com/21034990/179029073-98012bda-eda3-4578-b027-b7a7ba5b17c7.png"  width = 400>

