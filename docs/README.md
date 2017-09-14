# Overview

This is a relatively simple project which uses Canny edge detection and Hough line transforms to detect traffic lanes in static images and video. In this project, we use pure python code, relying heavily on OpenCV and Matplotlib to process and present the results. This project does NOT use machine learning techniques.

## Dependencies
If you need help setting up dependencies, or would like to mirror the dependencies used to develop this code, see  [mholzel.github.io/drive/](https://mholzel.github.io/drive/).

## Quick-Start
If you just want to see the core code of this project at work, then clone [this repo](https://github.com/mholzel/cannyHoughLanes) and run

    python lanes.py

This will run the lane detection code on the provided sample images and videos. Alternatively, you can open the Jupyter notebook which repeats much of the discussion here.


  Note: the test images and videos are in the `test_images` and `test_videos` folders, where the subfolders in those directories will contain the processed outputs. This repo comes packaged with the processed outputs already included, so if you want to check that the code is working on your system, you should delete the processed outputs and rerun the code. For instance, you can delete the folder `test_videos/outputs`, but you should not delete the videos in the root folder `test_videos/`.

---

## Finding Lanes
The goal of this project is to develop a pipeline for traffic lane identification which takes road images from a video as input, and returns a video output, where the lanes have been highlighted.


### Pipeline

Given a color image like

<img src="https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Original.png" alt="Original Image" style="width: 100%;"/>  

we use the following process to detect traffic lanes:

1. Convert the image to grayscale:    
<img src="https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Grayscale.png" alt="Grayscale Image" style="width: 100%;"/>

1. Apply a Gaussian blur filter. Raw images (particularly jpgs) tend to have a lot of noise in them, which we should try to attenuate. Although we can't perfectly remove all noise artifacts, passing the image through a Gaussian blur filter  will attenuate such effects:  
<img src="https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Blurred.png" alt="Blurred Image" style="width: 100%;"/>

1. Use Canny edge detection to find edges. The lane markers should have a strong contrast with the background road (if the lane markers are clearly visible). Hence we can start the lane detection process by focusing on places in the image where we see a high contrast between adjacent pixels. A Canny filter locates such places:  
<img src="https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Canny.png" alt="Canny Image" style="width: 100%;"/>

1. As we can see in the previous image, the lane markers are not the only *edges* in an image. Anything in an image that produces a high contrast with the background might be picked up as an edge, such as the outline of another car. Hence we need to limit our search for edges to the places where we actually expect to see lane markers. In our case, this is the triangle connecting the image's bottom left, bottom right, and center pixels:   
<img src="https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Trimmed.png" alt="Trimmed Image" style="width: 100%;"/>

1. The Canny filter only highlights edges in an image for us by setting their pixel value to *white*. However, since we expect the lane markers  to be straight lines (at least in the near-field range), we want to extract lines from this image. The Hough line transform does exactly that, although we have to tune many parameters which define what constitutes a *line*. The following figure shows all of the detected Hough lines, where each line is represented by a different color:  
<img src="https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Hough.png" alt="Hough Image" style="width: 100%;"/>

1. Unfortunately, the Hough line transform isn't going to just return everything we want. Instead of long lines denoting the left and right lanes, it tends to leave us with many small lines, which we will later need to average to estimate the left and right lane boundary. Hence at this point, we need to filter out any Hough lines that are clearly noise. To us, this means excluding lines which are approximately horizontal. For instance, any line with an absolute slope less than 0.3 is probably not part of a lane marker, so we remove it. (Note that such erroneous lines tend to be short, so it may be hard to see the change between this step and the previous figure):  
<img src="https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/FilteredHough.png" alt="Filtered Hough Image" style="width: 100%;"/>

1. Finally, with our Hough line results filtered, we separate the lines into those which likely belong to the left and right lane by computing their slopes. Specifically, if we imagine the y-axis as increasing toward the top of the page, we can roughly say that lines with a positive slope are likely part of the left lane boundary, and those with a negative slope are likely part of the right lane boundary. Once we have made this separation, we compute the weighted average of the slopes and y-axis intercepts of the lines for each lane, where the averages are weighted by the line lengths. This leaves us with a single line denoting the left lane and a single line denoting the right lane:  
<img src="https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Lanes.png" alt="Annotated Image" style="width: 100%;"/>

### Testing
Sample images and videos are packaged with this repo in the `test_images` and `test_videos` directories. You can run the test suite by either calling

    python lanes.py

or by running the `test()` method in the `lanes` module. When run with no arguments, this will look for images in the provided directories. You can add and remove images and videos from these directories, or (preferably) create new folders of test images and videos. For instance, if you have a directory of images in the directory `more_test_images`, then you can simply pass that as an argument to the `test` function:

    test( image_dir=`more_test_images/` )

This call will test new image directory and default video directory, but NOT the default image directory. You can think of this call as replacing the default image directory.

Finally, note that if you do not want to test either images or videos, then you can specify that directory as `None`. For instance,

- To test only images: `test( video_dir = None )`
- To test only videos: `test( image_dir = None )`
- To test nothing????: `test( image_dir = None, video_dir = None )`

### Processing Images and Videos
The core lane detection code is contained in the `lanes.detect` method, which takes an image and a set of `DetectionParameters` as inputs, and returns the processed image with annotated lanes as output. For instance, given the following image or its path:  

<img src="https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Original.png" alt="Original Image" style="width: 100%;"/>

the `lanes.detect` method returns the image:  

<img src="https://raw.githubusercontent.com/mholzel/cannyHoughLanes/master/test_images/solidWhiteCurve/Lanes.png" alt="Annotated Image" style="width: 100%;"/>

Note that we are displaying the images here, but the detect method does not automatically show them, it simply returns the raw data.

Furthermore, note that since the `lanes.detect` method is designed to work on a single static image, you can use it to process static images or video frames. Hence for convenience, we have provided two convenience methods for processing images and video, namely: `process_image` and `process_video`. The former requires either an image or a path to an image, while the latter requires the path to the input video you would like to process, and the path where you would like the output video saved. For instance, given the input video:

<video width="100%" controls>
  <source src="https://github.com/mholzel/cannyHoughLanes/raw/master/test_videos/solidWhiteRight.mp4" type="video/mp4">
  Your browser does not support HTML5 video.
</video>


the method `process_video` produces the following output when called with all of the default arguments:


<video width="100%" controls>
  <source src="https://github.com/mholzel/cannyHoughLanes/raw/master/test_videos/output/solidWhiteRight.mp4" type="video/mp4">
  Your browser does not support HTML5 video.
</video>

### Documentation
The documentation for this project can be found here.

### Issues and Future Work
At its core, this pipeline was developed to work on static images, that is, the identification of lanes at one point is time is completely independent from the identification of lanes at another point in time. Although this makes the code simple, this will clearly present a problem for stretches of road where the lane markings are either poorly visible or not present. In such circumstances, it would be preferable to somehow utilize the lanes identified in previous frames to determine the lane locations. For instance, one could easily imagine estimating the lane marking in a frame by computing a weighted average with the estimates from the past 5 frames to *smooth out* some of the jitter that is clearly visible in the videos.
