# OCR Using CNN
In this project, I have built, trained, analyzed, modified a Convolutional Neural Network CNN to learn recognizing numbers and greek letters.

The end goal of the system is to recognise handwrittings, with system tunning to achieve 90% accuracy.

### Method

To prepare a real life testing dataset, as shown below the capture at the left, I pre-processed my own set of handwrittings into binary images.  This is to match with the format of the MNIST dataset. 

At right upper corner, it shows the prediction result from the CNN before tunning.  As we can see the from the value of "prediction", about 60% of the characters were predicted correctly.  

![image](https://user-images.githubusercontent.com/21034990/176381947-5a45a6b7-511a-4099-8e65-5be10de0ca08.png)

### Main codes 
Q1a - 1e    Q1a - CnnCoreStructure.py<br>
Q1f - 1g    Q1f - Read trained MNIST network3.py<br>
Q2          Q2a - Examine network.py<br>
Q3          Q3 Create a digit embedding.py<br>
Q4i         Q4i - Design your own experiment Epoch.py<br>
Q4ii        Q4ii - Design your own experiment batch size.py<br>
Q4iii       Q4iii - Design your own experiment batch normalization.py<br>

1. The MNIST digit data consists of a training dataset of 60k 28x28 labeled digits and a testing dataset of 10k 28x28 labeled digits. The dataset was imported directly from the torchvision package as torchvision.datasets.MNIST.   Below capture shows the examples of the MNIST digit dataset with the corresponding ground truth.

<img src = "https://user-images.githubusercontent.com/21034990/177019922-2f674cf3-daf6-44cd-9e23-1e7fea3aa37c.png" width = 400>

2. Built the network with 2 convolution layers with max pooling and dropout, then trained the model for 5 epochs with batch size = 64.  From the plot below, we can see when the system have been trained with 60k examples, meaning almost 1 epoch, that the negative log likelihood loss has significantly dropped from more than 2 to reach a plateau at surrounding the 0.5 of the cost function.

<img src = "https://user-images.githubusercontent.com/21034990/177019931-3d9b189b-c1cb-46df-9703-2f54866ce848.png" width = 400>

From the network output as shown, we can further visualize the network has shown significant improvement at the second epoch.

```
Test set: Avg. loss: 2.3065, Accuracy: 1101/10000 (11%)

Test set: Avg. loss: 0.2158, Accuracy: 9337/10000 (93%)

Test set: Avg. loss: 0.1350, Accuracy: 9596/10000 (96%)

Test set: Avg. loss: 0.1036, Accuracy: 9655/10000 (97%)

Test set: Avg. loss: 0.0929, Accuracy: 9701/10000 (97%)

Test set: Avg. loss: 0.0786, Accuracy: 9754/10000 (98%)
```

The system is able to classify all 10 examples correctly after training.

<img src = "https://user-images.githubusercontent.com/21034990/177020029-8ffe6900-d00a-4c49-afb5-23c699d0652e.png" width = 400>

3.  Now, time to see if machine can tell the meaning of my handwrittings.  I fed my own handwritings to see how well the system predict/classify.  

The challenge to the prediction is, my own handwritting images were scaled down from 1k to 28x28, so despite it was written fairly thick at a whiteboard, the scaled down images are still not as strong in intensity as the MNIST dataset.

So the result of prediction can just achieve 60% accuracy, but we will see later on it could be fixed.

<img src = "https://user-images.githubusercontent.com/21034990/177020056-a88ca893-9d17-4ab5-b7df-ad99cf590942.png" width = 400>

4.Neural network is well known as black boxes.  Below I am trying to examine convolutional network and analysis the first convolutional layer

<img src = "https://user-images.githubusercontent.com/21034990/177020072-10b6dbf9-f0a9-496c-bba3-73c386a161a4.png" width = 400>

By observing the filters, it looks like the 5th filters (from left to right) is ridge detection as the boundary filters are all big negatives in value (black).

<img src = "https://user-images.githubusercontent.com/21034990/177020089-9992e96c-71d5-4d92-9b80-3900813899e2.png" width = 400>

To observe the second convolutional layer, I have built a truncated model.  The sixth picture looks like sobel y filter.

<img src = "https://user-images.githubusercontent.com/21034990/177020099-71c2df2a-5189-407a-ae87-1907d8aff43b.png" width = 400>

5.Now, using the same weights learnt from the MNIST dataset, I deployed an embedding with the network - trained the network with the same convolutional layers, then instead of passing to dense layers, i took out the vectors and passed them to my own KNN classifier.  By such, I can use my own choice of classifiers/predictors. (so now the network only has 2 convolutional layers with no output layers)

And now, instead of feeding MINIST digits again, I trained the network with another 3 x 9 greek letters from professor, then tested with my own Greek letter handwrittings.

With my own Greek symbol, the predictions is about 2/3 correctness.

<img src = "https://user-images.githubusercontent.com/21034990/177020106-0deaa1c8-377d-4a21-bfd6-a938903f4b23.png" width = 400>

6. Back to step 3, we want to get a better prediction performance.  Therefore, I have experimented tuning hyper parameters with 
- different number of epochs(5, 10, 15, 20, 25, 30) 
- batch sizes 64, 128, 192, 256, 320, 384
- replaced 2 dropout layers with 2 batch normalisation layers

And run with this total 6 x 6 x 2(with batchnormal or with dropout) 72 variations/combinations.  The best parameters experimented, was 15 epochs or 256 as the batch size.

Now, all digits are classified correctly.

<img src = "https://user-images.githubusercontent.com/21034990/179029073-98012bda-eda3-4578-b027-b7a7ba5b17c7.png"  width = 400>

