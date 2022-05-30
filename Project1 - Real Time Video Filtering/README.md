## Description

In this project, I would be practicing the principal of what we have learnt about a pixel or an image is composed of, what is colour, and mathematical models for image processing/filtering, by understanding a pixel value.  We firstly start with greyscale filters to understand the concept of colour channels composition and weightings.  

I created a 5x5 Gaussian filter to blur an image for the next step.  Gaussian filter is a low pass filter, is very commonly used for facial retouching.  It takes a crucial role of isolating colour and tone from texture (high frequency).

To cartoonize an image, it took all together 5 filters.  The principle is finding edges by gradient magnitude, then on a quantised and blurred image, make the edges dark to simulate the effect of a cartoon.

In the extension part, I have implemented additional filter from scratch by inverting pixels from left to right to produce an mirrored image.  My ultimate goal was to create a kaleidoscope.  I have also add interactions between camera and player, by creating a meme generator, video recording, and Chinese new year overlay.

## Greyscale by cvtcolor

![image](https://user-images.githubusercontent.com/21034990/170897238-3bc8b1cd-3cb1-41d3-861f-171c725adc64.png)
![image](https://user-images.githubusercontent.com/21034990/170897246-43aacc55-2caf-4502-84fb-5a1a9a892574.png)

As depicted by cvtColor documentation from opencv.org, grayscale is weighted by 0.299 R + 0.587 G + 0.114 B. It could be observed that the green channel is lower in intensity (when compared with the greyscaling by copying a color channel below).

Greyscale by copying one channel to the others

 

To compare the effect of color channel weighting, at this task, I have chosen to copy B channel to other channels.  It can be seen the green channel is showing more intensity (less dark).  Also observed loss of quality by copying channels. 

Blur

 

Blurring effect by 5x5 Gaussian filter.

Gradient Magnitude

 

Applied Sobel x and Sobel y filter, then applied the result of the magnitude into gradient magnitude.

Quantize

 

 Quantised image by 15 levels and 5x5 Gaussian filter.

Cartoon

 

Cartoonization by magnitude threshold 20.

Others - Allow user adjusting Contrast

 

Filter : 1.2 * pixel value + 20.

Extension 1 - Mirror

 

By inverting pixel from left to write from scratch.

Extension 2 - Meme Generator

A meme generator with cin to accept the mood from player.

Extension 3 - Video recording (click at the pic to view the video)

Courtesy in video meetings/recording to let the player knows that it is currently recording.

Extension 4 - Overlay

 

Happy Chinese New year!  Extension 4 is to overlay 2 images.

Reflection

Computer vision is green field to me.  I learnt -

Basic C++

OpenCV

xcode

filtering concept, which is very fun

edge detection, i am sure it would be used most of the time later on.

Time management!

Acknowledgement 

I am really thankful to Professor Bruce Maxwell.  I had obstacles at understanding the file type and structure in order to proceed, this is my first c++ project, and professor was very supportive.  

I would also like to thank TA Aniket and Amit.  They shed with me light and they talked in a way that they hope us to understand the topics.

Materials - Murtaza's Workshop - Robotics and AI which helped me to install opencv and xcode to start with, and also opencv concept.

Codemy to start with C++






