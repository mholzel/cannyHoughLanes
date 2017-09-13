import matplotlib.pyplot as plt
import cv2, inspect, io, math, matplotlib, numpy, os, tqdm, urllib, zipfile

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
    Apply a mask to the image using a polygon with the specified vertices.
    Specifically, all points in the image outside of the polygon are changed to "black".
    """
    # Create a blank mask (that is, a black image). Using black is critical for the mask operation,
    # otherwise, the comparison at the end of the function will not work
    mask = numpy.zeros_like(img)

    # Fill all of the pixels inside the polygon defined by "vertices" with a non-black color
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Returning the image only where mask pixels are nonzero
    return cv2.bitwise_and(img, mask)


def draw_lines(img, lines=[], thickness=2):
    """
    This function draws `lines` with the specified `thickness` using a rainbow of colors.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, see <weighted_img>.
    """
    if isinstance(img, tuple):
        shape = img
        img = numpy.zeros((shape[0], shape[1], 3), dtype=numpy.uint8)
    colors = matplotlib.cm.rainbow(numpy.linspace(0, 1, len(lines)))
    for color, line in zip(colors, lines):
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), 255 * numpy.array(color[0:3]), thickness)
    return img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    '''
    This function computes probalistic Hough lines using OpenCV's "HoughLinesP"
    '''
    return cv2.HoughLinesP(img, rho, theta, threshold, numpy.array([]), minLineLength=min_line_len,
                           maxLineGap=max_line_gap)


def weighted_img(img, img0, alpha=0.8, beta=1., lam=0.):
    """
    This function overlays `img` on 'img0' using the following algorithm:

    output = img0 * alpha + img * beta + lambda

    NOTE: img0 and img must be the same shape!
    """
    return cv2.addWeighted(img0, alpha, img, beta, lam)


########################################
#
# Lane detection pipeline
#
########################################
class DetectionParameters(object):
    '''

    This class encapsulates most of the parameters that are used in the lane detection pipeline.

    :param rho: distance resolution in pixels of the Hough grid
    :param theta: angular resolution in radians of the Hough grid
    :param threshold: minimum number of votes (intersections in Hough grid cell)
    :param min_line_length: minimum number of pixels making up a line
    :param max_line_gap: maximum gap in pixels between connectable line segments
    :param min_slope: when detecting lanes, we look for hough lines. However, we should reject horizontal lines since /
        they are unlikely to be lanes. This is the minimum absolute slope a line must have NOT to be rejected.
    '''

    def __init__(self, blur=5, canny_low=50, canny_high=150, rho=1, theta=1 * numpy.pi / 180, threshold=10,
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
        self.min_slope = abs(min_slope)


def path_to_image(path):
    '''
    :param path: The path to an image
    :return: the image
    '''
    return matplotlib.image.imread(path)


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
    right_data = [si for si in data if si[0][0] > 0]
    left_data = [si for si in data if si[0][0] <= 0]

    # It might be a good idea to keep only the longest lines that we detected since short lines tend to be noise.
    # It you want to do that, fill in a positive number here. 0 means "select all".
    lines_to_keep = 0
    right_data = numpy.sort(right_data, axis=0)[-lines_to_keep:]
    left_data = numpy.sort(left_data, axis=0)[-lines_to_keep:]

    # Finally, computed the weighted mean of the slopes and intercepts
    right, right_len = zip(*right_data)
    left, left_len = zip(*left_data)

    right_mean = numpy.average(right, axis=0, weights=right_len)
    left_mean = numpy.average(left, axis=0, weights=left_len)
    return (right_mean, left_mean)


def draw_lanes(image, lanes, color=[255, 0, 0], thickness=15):
    '''
    This function draws the lanes on the specified image, where the lanes are specified as list of (slope,intercept) tuples.
    '''
    # We need the image height to calculate the intercept
    image = numpy.array(image, copy=True)
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
    '''
    This is the core lane detection algorithm, which detects lanes in a static image by

    1. Converting the image to grayscale
    2. Blurring the image to attenuate image artifacts and noise
    3. Applying Canny edge detection
    4. Applying a "region of interest map" to remove edges outside of a specified area
    5. Computing the Hough lines
    6. Removing the Hough lines which are vertical and those which are nearly horizontal (since these are unlikely to be lanes)
    7. Separating lines into left and right by examining whether their slope is positive or negative
    8. Computing the weighted average y-axis intercepts and slopes for the left and right lanes, where the weights are taken to be the line lengths.

    :param image: Either a 3-channel color image or the path to such an image.
    :param params: The DetectionParameters that should be used for lane detection.
    :param name: The name of the frame. This is useful particularly when testing multiple images.
    :param verbose: If true, then this will show all of the stages in the detection pipeline in one concise plot.
    :param processed_frame_path: The path of where to save the processed frame plot. If this is None, then the plot is not saved.
    :return: The processed image, with the lanes annotated.
    '''
    # If the input is a string, we assume it is a path and convert it to an image.
    if isinstance(image, str):
        name = image
        image = path_to_image(name)

    # At this point, "image" should be an image.
    grey = grayscale(image)
    blurred = gaussian_blur(grey, params.blur)
    canned = canny(blurred, params.canny_low, params.canny_high)
    vertices = numpy.array([[vertex(canned, .05, 1), vertex(canned, .5, .58), vertex(canned, .95, 1)]])
    trimmed = region_of_interest(canned, vertices)
    lines = hough_lines(trimmed, params.rho, params.theta, params.threshold, params.min_line_length,
                        params.max_line_gap)

    # Remove the extra dimension from the data to make processing easier
    lines = [line[0] for line in lines]
    hough = draw_lines(trimmed.shape, lines)
    filtered_lines = filter_lines(lines, params)
    filtered_hough = draw_lines(trimmed.shape, filtered_lines)
    lane_boundaries = separate_lines(filtered_lines, params)
    lanes = draw_lanes(image, lane_boundaries)

    # If you either want verbose output or if you want to save the processed frame pipeline, then we will generate the plot showing all of the steps in the pipeline.
    if verbose or (processed_frame_path is not None):
        print("drawing")
        if False:
            plt.imshow(lanes, cmap='gray')
            plt.axis('off')
        else:
            composite = False
            images = [(image, 'Original'), (grey, 'Grayscale'), (blurred, 'Blurred'), (canned, 'Canny'),
                      (trimmed, 'Trimmed'), (hough, 'Hough'), (filtered_hough, 'FilteredHough'), (lanes, 'Lanes')]
            n = math.floor(math.sqrt(len(images)))
            if composite:
                for (c, img) in enumerate(images):
                    plt.subplot(n, math.ceil(len(images) / n), c + 1)
                    plt.imshow(img[0], cmap='gray')
                    plt.axis('off')
                    plt.title(img[1])
                plt.suptitle(name)
                if processed_frame_path is not None:
                    plt.savefig(processed_frame_path, bbox_inches='tight', dpi=600)
            else:
                os.makedirs(processed_frame_path + "/", exist_ok=True)
                for (c, img) in enumerate(images):
                    plt.imshow(img[0], cmap='gray')
                    plt.axis('off')
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    if processed_frame_path is not None:
                        plt.savefig(processed_frame_path + "/" + img[1] + ".png", bbox_inches='tight', pad_inches=0,
                                    dpi=100)
        if verbose:
            plt.show(block=False)
    return lanes


def process_image(image, count=None, processed_frames_dir=None, params=DetectionParameters()):
    '''
    This function processes a single static image, returning a 3-channel processed color image as output using
    the <detect_lanes> method.
    '''
    if count is not None and processed_frames_dir is not None:
        os.makedirs(processed_frames_dir, exist_ok=True)
        processed_frame_path = processed_frames_dir + 'frame' + str(count)
    else:
        processed_frame_path = None
    return detect_lanes(image, params, processed_frame_path=processed_frame_path)


def process_video(input_path, output_path, frame_processor=process_image, fps=None):
    '''
    This function processes the video at the specified input path, saving the processed video at the specified output path.
    Specifically, each frame will be processed using the frame processor, and the output video will be saved at the specified fps.
    :param frameProcessor: a function which should take a frame as input, and returned the processed frame as output. /
    If this function accepts more than one parameter, then the cumulative frame count will be passed as a second input.
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

    # Set up the progress bar and grab the fps if one was not specified
    frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    progressbar = tqdm.tqdm(total=frames)
    if fps is None:
        fps = input_video.get(cv2.CAP_PROP_FPS)
        print("Using " + str(fps) + "fps")

    # Now, for each frame of the video, run the frame processor and save the output in a new video
    while input_video.isOpened():
        success, frame = input_video.read()
        if (not success) or (cv2.waitKey(10) & 0xFF == ord('q')):
            break
        if output_video is None:
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        count += 1
        try:
            if pass_count:
                output_video.write(frame_processor(frame, count))
            else:
                output_video.write(frame_processor(frame))
        except:
            output_video.write(frame)
        progressbar.update()
    progressbar.update(n=(frames - progressbar.n))
    progressbar.close()
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

    Note: this function will create the output_path directory if it does not exist.
    '''

    # Make sure that the specified output directory already exists
    os.makedirs(output_path, exist_ok=True)

    # Open the video and save each of the frames
    input_video = cv2.VideoCapture(input_path)
    count = 0
    while input_video.isOpened():
        success, frame = input_video.read()
        if (not success) or (cv2.waitKey(10) & 0xFF == ord('q')):
            break
        count += 1
        matplotlib.image.imsave(output_path + 'frame' + str(count) + '.jpg', frame)
    cv2.destroyAllWindows()
    input_video.release()


def is_dir(dir):
    ''' Determine whether the specified directory exists and is a directory. '''
    return os.path.isdir(dir)


def is_empty(dir):
    '''
    Determine whether the specified directory is empty. Note that you should check whether the directory exists before
    this call.
    '''
    return not os.listdir(dir)


def is_not_none_and_empty(dir):
    return (dir is not None) and is_empty(dir)


########################################
#
# Testing
#
########################################
def test(image_dir="test_images/", video_dir="test_videos/", params=DetectionParameters()):
    '''
    Run the lane detection code on all of the images in the image directory and videos in the video directory.
    If a directory is specified as None, then those tests will not be run. For instance:

    To only test the detection on images, call "test( video_dir = None )"
    To only test the detection on videos, call "test( image_dir = None )"

    Note that if you specify a directory which turns out to be empty, we will populate it with images or videos from
    https://github.com/mholzel/drive/files/1296126/data.zip

    :return:
    '''
    # Depending on the project configuration, you may need to move to the directory of this script
    os.chdir(os.path.dirname(__file__))

    # # I would prefer not to package the images and videos in the repo, but I haven't got around to fixing this yet.
    # # If either the image or video directory is not None and is empty, then we will download and unpack the standard test files
    # if (not is_dir(image_dir) or is_not_none_and_empty(image_dir)) or (
    #             not is_dir(video_dir) or is_not_none_and_empty(video_dir)):
    #     with urllib.request.urlopen("https://github.com/mholzel/drive/files/1296126/data.zip") as url:
    #         print("Start")
    #         content = url.read()
    #         print("content: ", content)
    #         ior = io.StringIO(content)
    #         print("io", ior)
    #         zip = zipfile.ZipFile(ior)
    #         print("zip", zip)
    #         for name in zip.namelist():
    #             print(name)
    # return

    # Run the tests for all of the test images if the directory was not specified as None
    if image_dir is not None:

        # Now apply lane detection to each test image, skipping directories
        for image_path in os.listdir(image_dir):
            if os.path.isdir(image_dir + image_path):
                continue
            processed_frame_path = image_dir + os.path.splitext(image_path)[0]
            print(processed_frame_path)
            detect_lanes(image_dir + image_path, params, processed_frame_path=processed_frame_path)

    # Run the tests for all of the test videos if the directory was not specified as None
    if video_dir is not None:

        # Now we will work with video. Try to detect the lanes in each of the videos. The processed video will be
        # saved in the following output directory
        output_dir = video_dir + 'output/'
        os.makedirs(output_dir, exist_ok=True)

        # Furthermore, you can optionally save the frames of the video as well as the processedFrames for each video
        save_frames = False
        output_frame_dir = video_dir + 'frames/'
        save_processed_frames = False
        output_processed_frame_dir = 'processed_frames/'

        # Furthermore, optionally, you can save the frames of the video as well as the processedFrames for each video
        for video in os.listdir(video_dir):

            print(video)

            # Define the path to the video we want to process, as well as where we should save the processed video
            input_path = video_dir + video
            output_path = output_dir + video

            # It is sometimes also useful to save the video frames of the original video in case you want to test
            # why individual frames are not giving the desired behavior
            if save_frames:
                save_video_frames(input_path, output_frame_dir + video + '/')

            # If you want to save the individual processed frames, then you can specify a directory here.
            # Otherwise, if you do not want to save the individual frames, then set this value to None.
            if save_processed_frames:
                processed_frames_dir = output_processed_frame_dir + video + '/'
            else:
                processed_frames_dir = None
            process_video(input_path, output_path,
                          lambda x, count: process_image(x, count=count, processed_frames_dir=processed_frames_dir))


########################################
#
# Testing
#
########################################
if __name__ == "__main__":
    test(video_dir = None)
