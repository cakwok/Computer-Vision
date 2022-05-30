## Description

Hello world to computer vision!  In this project, the aim is to apply filters to live stream video pixel by pixel by C++. 

I have built Guassian filter from scratch, created grayscale live video, applied gradient magnitude to locate edges, quantized a live video with color thresholdings, cartoonization, mirroring an image, contrast, meme generation and video recording, and recorded result as below.

## Greyscale by cvtcolor

<img src = "https://user-images.githubusercontent.com/21034990/170897238-3bc8b1cd-3cb1-41d3-861f-171c725adc64.png" width=400> <img src = "https://user-images.githubusercontent.com/21034990/170897246-43aacc55-2caf-4502-84fb-5a1a9a892574.png" width=400>

As depicted by cvtColor documentation from opencv.org, grayscale is weighted by 0.299 R + 0.587 G + 0.114 B. It could be observed that the green channel is lower in intensity (when compared with the greyscaling by copying a color channel below).

## Greyscale by copying one channel to the others

![image](https://user-images.githubusercontent.com/21034990/170897490-2c4730e9-913d-46b5-a5c1-a4e6adec1073.png)
 
To compare the effect of color channel weighting, at this task, I have chosen to copy B channel to other channels.  It can be seen the green channel is showing more intensity (less dark).  Also observed loss of quality by copying channels. 

## Blur
![image](https://user-images.githubusercontent.com/21034990/170897538-9b3334a0-32af-400a-8e2e-24065c2766ac.png)

Blurring effect by 5x5 Gaussian filter.

## Gradient Magnitude
![image](https://user-images.githubusercontent.com/21034990/170897554-99bc4273-aedf-4e39-a991-517e7455d2fa.png)

Applied Sobel x and Sobel y filter, then applied the result of the magnitude into gradient magnitude.

## Quantize
![image](https://user-images.githubusercontent.com/21034990/170897613-b8e271f5-a4a7-4202-ad72-7ffbeb675756.png)

Quantised image by 15 levels and 5x5 Gaussian filter.

## Cartoon
![image](https://user-images.githubusercontent.com/21034990/170897629-d1c16747-ee9e-4c90-97e5-2f80d3b0a1c9.png)

Cartoonization by magnitude threshold 20.

## Others - Allow user adjusting Contrast
![image](https://user-images.githubusercontent.com/21034990/170897654-c995e138-1a8c-46f4-bc3a-4296f5bc166a.png)

Filter : 1.2 * pixel value + 20.

## Extension 1 - Mirror
![image](https://user-images.githubusercontent.com/21034990/170897677-390c2345-38a1-464c-9e89-7584f566db25.png)

By inverting pixel from left to write from scratch.

## Extension 2 - Meme Generator
![image](https://user-images.githubusercontent.com/21034990/170897690-837d4ce1-e33f-4b0c-bd6b-aafe69e7d276.png)

A meme generator with cin to accept the mood from player.

## Extension 3 - Video recording (click at the pic to view the video)
![image](https://user-images.githubusercontent.com/21034990/170897721-6db29e7e-ba9f-4263-82ef-a914b793d7fc.png)

Courtesy in video meetings/recording to let the player knows that it is currently recording.

## Extension 4 - Overlay
![image](https://user-images.githubusercontent.com/21034990/170897733-a4d96e5c-88df-40e0-bcec-5cfb1a16a7c8.png)

Happy Chinese New year!  Extension 4 is to overlay 2 images.
