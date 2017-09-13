# Overview

This is a relatively simple project which uses Canny edge detection and Hough line transforms to detect traffic lanes in static images and video. In this project, we use pure python code, relying heavily on OpenCV and Matplotlib to process and present the results. This project does NOT use machine learning techniques.

## Dependencies
If you need help setting up dependencies, or would like to mirror the dependencies used to develop this code, read the README at  [mholzel.github.io/drive/](https://mholzel.github.io/drive/).

## Quick-Start
If you just want to see the core code of this project at work, then clone [this repo](https://github.com/mholzel/cannyHoughLanes) and run

    python lanes.py

This will run the lane detection code on the provided sample images and videos.

---

## Finding Lanes
The goal of this project is to develop a pipeline for traffic lane identification which takes road images from a video as input, and returns a video output, where the lanes have been highlighted.


### Testing
Sample images and videos are packaged with this repo in the `test_images` and `test_videos` directories. You can run the test suite by either calling

    python lanes.py

or by running the `test()` method in the `lanes` module. When run with no arguments, this will look for images in the provided directories. You can add and remove images and videos from these directories, or (preferably) create new folders of test images and videos. For instance, if you have a directory of images in the directory `more_test_images`, then you can simply pass that as an argument to the `test` function:

    test( image_dir=`more_test_images/` )

This call will test new image directory and default video directory, but NOT the default image directory. You can think of this call as replacing the default image directory.

Finally, note that if you do not want to test either images or videos, then you can specify that directory as `None`. For instance,

1. To test only images: `test( video_dir = None )`
1. To test only videos: `test( image_dir = None )`
1. To test nothing????: `test( image_dir = None, video_dir = None )`

### Pipeline

Given an image
![alt text][original]
![alt text][gray]
![alt text][blurred]
![alt text][canny]
![alt text][trimmed]
![alt text][hough]

[original] https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Original.png "Original Image"
[gray] https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Grayscale.png "Grayscale"
[blurred] https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Blurred.png "Blurred"
[canny] https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Canny.png "Canny"
[trimmed] https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Trimmed.png "Trimmed"
[hough] https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Hough.png "Hough"
[filteredHough] https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/FilteredHough.png "Hough"

### Issues and Future Work
At its core, this pipeline was developed to work on static images, that is, the identification of lanes at one point is time is completely independent from the identification of lanes at another point in time. Although this makes the code simple, this will clearly present a problem for stretches of road where the lane markings are either poorly visible or not present. In such circumstances, it would be preferable to somehow utilize the lanes identified in previous frames to determine the lane locations. For instance,
