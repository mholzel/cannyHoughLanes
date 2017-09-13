# **Lane detection using Canny edge detection and Hough line transforms**

## Overview
This is a relatively simple project which uses Canny edge detection and Hough line transforms to detect traffic lanes in static images and video. In this project,
we use pure python code, relying heavily on OpenCV and Matplotlib to process and
present the results. This project does NOT use machine learning techniques.

## Dependencies
If you need help setting up dependencies, or would like to mirror the dependencies
used to develop this code, read the README at
https://github.com/mholzel/drive.

### Quick-Start
If you just want to see the core code of this project at work, then clone this repo and run

    python lanes.py

This will download sample images and videos from https://github.com/mholzel/drive/files/1296126/data.zip and run the lane detection code.

---

## Finding Lanes
The goal of this project is to develop a pipeline for traffic lane identification which takes road images from a video as input, and returns a video output, where the lanes have been highlighted.


#### Test Data
Sample images and videos are available at https://github.com/mholzel/drive/files/1296126/data.zip. You can either
1. clone unzip  
1. clone unzip  
1. clone unzip  

#### Pipeline Description


### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I ....

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image:

![alt text][image1]


### Issues and Future Work
At its core, this pipeline was developed to work on static images, that is, the identification of lanes at one point is time is completely independent from the identification of lanes at another point in time. Although this makes the code simple, this will clearly present a problem for stretches of road where the lane markings are either poorly visible or not present. In such circumstances, it would be preferable to somehow utilize the lanes identified in previous frames to determine the lane locations. For instance,
