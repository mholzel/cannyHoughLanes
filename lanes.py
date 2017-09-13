import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2, math, os
import inspect

'''
This is a relatively simple project which uses Canny edge detection and Hough line transforms to detect traffic lanes in 
static images and video. In this project, we use pure python code, relying heavily on OpenCV and Matplotlib to process and
present the results. This project does NOT use machine learning techniques. To see a simple demo of this 
module at work, just try running this function:

python lanes.py

This will download some test images and test data, and run the lane detection algorithm.
'''


def grayscale(img):
    """
    Convert an image to grayscale. This will return an image with only one color channel.
    Note that to properly view this image in matplotlib, you should use the 'gray' 
    colormap, that is, you can convert an image 'img' to grayscale and view it using 
    
    matplotlib.pyplot.imshow(grayscale(img),cmap='gray')
    """
    # If the input image was read using cv2.imread(), then uncomment the following line 
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform using the specified thresholds. """
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel of the specified size"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Apply a mask to the image, converting all points in the image outside 
    of the polygon defined by the vertices to "black".
    """
    # Create a blank mask 
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines=[], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if isinstance(img, tuple):
        shape = img
        img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(lines)))
    for color, line in zip(colors, lines):
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), 255 * np.array(color[0:3]), thickness)
    return img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                           maxLineGap=max_line_gap)


# Python 3 has support for cool math symbols. NOTE: Does not work on Windows 10 with my Anaconda 3.6 distro

def weighted_img(img, initial_img, alpha=0.8, beta=1., lam=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + lambda
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lam)


########################################
#
# Lane detection pipeline
#
########################################
class DetectionParameters(object):
    '''

    This class encapsulates all of the parameters that are used. We do this so that everything is configurable in
    one place

    :param rho: distance resolution in pixels of the Hough grid
    :param theta: angular resolution in radians of the Hough grid
    :param threshold: minimum number of votes (intersections in Hough grid cell)
    :param min_line_length: minimum number of pixels making up a line
    :param max_line_gap: maximum gap in pixels between connectable line segments
    :param min_slope: when detecting lanes, we look for hough lines. However, we should reject horizontal lines.
    This is the minimum absolute slope a line must have NOT to be rejected.
    '''

    def __init__(self, blur=5, canny_low=50, canny_high=150, rho=1, theta=1 * np.pi / 180, threshold=10,
                 min_line_length=20,
                 max_line_gap=2, min_slope=.3):
        self.blur = blur
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.min_slope = min_slope


def path_to_image(path):
    '''
    :param path: The path to an image
    :return: the image
    '''
    return mpimg.imread(path)


def vertex(image, width_percent, height_percent):
    '''
    :param image:
    :param width_percent:
    :param height_percent:
    :return: The vertex in the specified image specified as a proportion of the width and height.
    For instance, if the specified image is (500x1000) (that is, width x height), and this function is called with
    ( image, .1, .9 ), then the returned tuple is (.1*500,.9*1000)=(50,900). Note that the output is rounded to the
    nearest integer.
    '''
    return (round(width_percent * image.shape[1]), round(height_percent * image.shape[0]))


def filter_lines(lines, params):
    '''
    This function removes all of the vertical and nearly horizontal lines since they are unlikely to be lanes
    '''
    # Remove the vertical lines
    lines = [line for line in lines if line[2] != line[0]]

    # Compute the slopes of the lines
    slopes = [(line[3] - line[1]) / (line[2] - line[0]) for line in lines]

    # Remove approximately horizontal lines
    return [sl[1] for (i, sl) in enumerate(zip(slopes, lines)) if abs(sl[0]) > params.min_slope]


def separate_lines(lines, params):
    '''
    The specified lines should be separable into left and right.
    We accomplish this using the following process:
    1) Divide the dataset into two groups: those with positive and negative slopes
    2) Return the weighted mean slope and intercept of each group, where the weighting is based on the line length
    '''
    # Compute the slopes of the lines
    slopes = [(line[3] - line[1]) / (line[2] - line[0]) for line in lines]

    # Compute the intercepts and line lengths
    intercepts = [line[3] - slope * line[2] for (line, slope) in zip(lines, slopes)]
    lengths = [math.sqrt((line[3] - line[1]) ** 2 + (line[2] - line[0]) ** 2) for line in lines]

    # Now we have a dataset of slopes, intercepts, and lengths
    data = list(zip(zip(slopes, intercepts), lengths))

    # Split the dataset into those with positive and negative slope
    rightData = [si for si in data if si[0][0] > 0]
    leftData = [si for si in data if si[0][0] <= 0]

    # It might be a good idea to keep only the longest lines that we detected since short lines tend to be noise.
    # It you want to do that, fill in a positive number here. 0 means "select all".
    lines_to_keep = 0
    rightData = np.sort(rightData, axis=0)[-lines_to_keep:]
    leftData = np.sort(leftData, axis=0)[-lines_to_keep:]

    # Finally, computed the weighted mean of the slopes and intercepts
    right, rightLen = zip(*rightData)
    left, leftLen = zip(*leftData)

    rightMean = np.average(right, axis=0, weights=rightLen)
    leftMean = np.average(left, axis=0, weights=leftLen)
    return (rightMean, leftMean)


def draw_lanes(image, lanes, color=[255, 0, 0], thickness=15):
    '''
    This function draws the lanes on the specified image, where the lanes are specified as (slope,intercept) tuples.
    '''
    # We need the image height to calculate the intercept
    image = np.array(image, copy=True)
    height = image.shape[0]
    y1 = int(round(height))
    y2 = int(round(0.6 * height))
    for lane in lanes:
        # We need to calculate the point where the lane would intercept the bottom of the screen (the image height)
        x1 = int(round((y1 - lane[1]) / lane[0]))
        x2 = int(round((y2 - lane[1]) / lane[0]))
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def detect_lanes(image, params, name='', verbose=False, processed_frame_path=None):
    # If the input is a string, we assume it is a path and convert it to an image.
    if isinstance(image, str):
        name = image
        image = path_to_image(name)

    # At this point, "image" should be an image.
    grey = grayscale(image)
    blurred = gaussian_blur(grey, params.blur)
    canned = canny(blurred, params.canny_low, params.canny_high)
    vertices = np.array([[vertex(canned, .05, 1), vertex(canned, .5, .58), vertex(canned, .95, 1)]])
    trimmed = region_of_interest(canned, vertices)
    lines = hough_lines(trimmed, params.rho, params.theta, params.threshold, params.min_line_length,
                        params.max_line_gap)

    # Remove the extra dimension from the data to make processing easier
    lines = [line[0] for line in lines]
    hough = draw_lines(trimmed.shape, lines)
    filtered_lines = filterLines(lines, params)
    filtered_hough = draw_lines(trimmed.shape, filtered_lines)
    lane_boundaries = separateLines(filtered_lines, params)
    lanes = draw_lanes(image, lane_boundaries)
    if verbose or (processed_frame_path is not None):
        if False:
            plt.imshow(lanes, cmap='gray')
        else:
            images = [(image, 'Original'), (grey, 'Grayscale'), (blurred, 'Blurred'), (canned, 'Canny'),
                      (trimmed, 'Trimmed'), (hough, 'Hough'), (filtered_hough, 'Filtered Hough'), (lanes, 'Lanes')]
            n = math.floor(math.sqrt(len(images)))
            for (c, img) in enumerate(images):
                plt.subplot(n, math.ceil(len(images) / n), c + 1)
                plt.imshow(img[0], cmap='gray')
                plt.title(img[1])
            plt.suptitle(name)
        if verbose:
            plt.show()
        if processed_frame_path is not None:
            plt.savefig(processed_frame_path, bbox_inches='tight', dpi=600)
    return lanes


def process_image(image, count=None, processed_frames_dir=None, params = DetectionParameters()):
    '''
    This function uses the specified DetectionParameters to search for lanes in the specified image. 
    
    '''
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)
    
    if count is not None and processed_frames_dir is not None:
        os.makedirs(processed_frames_dir, exist_ok=True)
        processed_frame_path = processed_frames_dir + 'frame' + str(count)
    else:
        processed_frame_path = None
    return detectLanes(image, params, verbose=False, processed_frame_path=processed_frame_path)


def process_video(input_path, output_path, frame_processor, fps=30):
    '''
    This function processes the video at the specified input path, saving the processed video at the specified output path.
    Specifically, each frame will be processed using the frame processor, and the output video will be saved at the specified fps.
    :param frameProcessor: a function which should take a frame as input, and returned the processed frame as output. If this function accepts more than one parameter, then the cumulative frame count will be passed as a second input.
    '''
    input_video = cv2.VideoCapture(input_path)
    output_video = None
    count = 0

    # Check whether the frame_processor takes one or two arguments. 
    # If it takes two args, then we will pass it the frame count as the second arg
    if len(inspect.getfullargspec(frame_processor).args) > 1:
        pass_count = True
    else:
        pass_count = False
        
    # Now, for each frame of the video, run the frame processor and save the output in a new video
    while input_video.isOpened():
        success, frame = input_video.read()
        if (not success) or (cv2.waitKey(10) & 0xFF == ord('q')):
            break
        if output_video is None:
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        count += 1
        print("frame: ", count)
        try:
            if pass_count:
                output_video.write(frame_processor(frame, count))
            else:
                output_video.write(frame_processor(frame))
        except:
            output_video.write(frame)
    cv2.destroyAllWindows()
    input_video.release()
    if output_video is not None:
        output_video.release()


def save_video_frames(input_path, output_path):
    '''
    This function takes the input path to a video, and then saves all of the video frames to the specified output path.
    You will typically use this function if an algorithm is failing on a particular video frame, and you 
    want to diagnose why. Specifically, in that case, you can save all of the video frames to file, and then 
    repeatedly call your detection algorithm only on that single frame. 
    '''
    
    # Make sure that the specified output directory already exists 
    os.makedirs(output_path, exist_ok=True)

    # Open the video and save each of the frames 
    input = cv2.VideoCapture(input_path)
    count = 0
    while input.isOpened():
        success, frame = input.read()
        if (not success) or (cv2.waitKey(10) & 0xFF == ord('q')):
            break
        count += 1
        mpimg.imsave(output_path + 'frame' + str(count) + '.jpg', frame)
    cv2.destroyAllWindows()
    input.release()


########################################
#
# Testing
#
########################################
if __name__ == "__main__":

    # Depending on the project configuration, you may need to move to the directory of this script
    os.chdir(os.path.dirname(__file__))

    # Define the directory where the test images live and test them.
    dir = "test_images/"
    params = DetectionParameters()
    for imagePath in os.listdir(dir):
        detectLanes(dir + imagePath, params, verbose=False)

    # Now we will work with video. Try to detect the lanes in each of the videos.
    # Furthermore, optionally, you can save the frames of the video as well as the processedFrames for each video
    input_dir = "test_videos/"
    output_dir = "test_videos_output/"
    output_frame_dir = 'test_videos_frames/'  # Raw video frames
    output_processed_frame_dir = 'test_videos_processed_frames/'  #
    os.makedirs(output_dir, exist_ok=True)
    save_frames = False
    save_processed_frames = False
    for video in os.listdir(input_dir):

        print(video)

        # Define the path to the video we want to process, as well as where we should save the processed video
        input_path = input_dir + video
        output_path = output_dir + video

        # It is sometimes also useful to save the video frames of the original video in case you want to test
        # why individual frames are not giving the desired behavior
        if save_frames:
            save_video_frames(input_path, output_frame_dir + video + '/')

        # If you want to save the individual processed frames, then you can specify a directory here.
        # Otherwise, if you do not want to save the individual frames, then set this value to None.
        if save_processed_frames:
            processedFramesDir = output_processed_frame_dir + video + '/'
        else:
            processedFramesDir = None
        processVideo(input_path, output_path, lambda x, count: process_image(x, count=count, processedFramesDir=processedFramesDir))
