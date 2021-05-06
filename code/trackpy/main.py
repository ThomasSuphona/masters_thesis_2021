import cv2
import numpy as np
import os
import glob
import time
from PIL import Image
import PIL.ImageOps as pimo
import matplotlib.pyplot as plt



def video_to_filtered_images(input_loc, output_loc, balance=0.0, dim2=None, dim3=None):

    #Parameters to undistort from fisheye
    DIM = (1920, 1080)
    K = np.array([[995.9674795793435, 0.0, 980.9711898019134],
                  [0.0, 997.8580870545059, 525.876314273917],
                  [0.0, 0.0, 1.0]])

    D = np.array([[0.039014117003568535],
                  [-0.13564797917630375],
                  [0.19245112256088762],
                  [-0.0946605817791271]])

    #Video to frames
    try:
        os.mkdir(output_loc)
    except OSError:
        pass


    # Log the time
    time_start = time.time()

    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)

    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)

    count = 0
    print("Converting video..\n")

    # Start converting the video
    while cap.isOpened():

        # Extract the frame
        ret, img = cap.read()

        # Undistort part
        dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
        assert dim1[0] / dim1[1] == DIM[0] / DIM[
            1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if not dim2:
            dim2 = dim1
        if not dim3:
            dim3 = dim1
        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
        # This is how scaled_K, dim2 and balance are used to determine the final K used to
        # un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3),
                                                                       balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        #Do invert and greyscale
        invert_img = (255-undistorted_img)
        grey_img = cv2.cvtColor(invert_img, cv2.COLOR_BGR2GRAY)
        cropped_img = grey_img[195:820, 592:1340]
        #cropped_img = grey_img[77:978, 435:1485]

        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count + 1), cropped_img)
        count = count + 1
        print('%d/%d frames processed'%(count, video_length))

        # If there are no more frames left
        if (count > (video_length - 1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            print("It took %d seconds for conversion." % (time_end - time_start))
            break

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def calibrate_crop(input_loc, output_loc, balance=0.0, dim2=None, dim3=None):
    # Parameters to undistort from fisheye
    DIM = (1920, 1080)
    K = np.array([[995.9674795793435, 0.0, 980.9711898019134],
                  [0.0, 997.8580870545059, 525.876314273917],
                  [0.0, 0.0, 1.0]])

    D = np.array([[0.039014117003568535],
                  [-0.13564797917630375],
                  [0.19245112256088762],
                  [-0.0946605817791271]])

    img = cv2.imread(input_loc)

    # Undistort part
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert dim1[0] / dim1[1] == DIM[0] / DIM[
        1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to
    # un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3),
                                                                   balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Do invert and greyscale
    invert_img = (255 - undistorted_img)
    grey_img = cv2.cvtColor(invert_img, cv2.COLOR_BGR2GRAY)
    #cropped_img = grey_img[77:978, 435:1485]
    cropped_img = grey_img[210:820, 580:1380]

    cv2.imwrite(output_loc + "/cropped.jpg", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

input_loc = '../input_videos/C30B838P3M.mp4'
output_loc = '../image_sequences/C30B838P3M/'

test_input = '../calibrate_crop/test_input/test.jpg'
test_output = '../calibrate_crop/test_output/'

video_to_filtered_images(input_loc, output_loc, balance=0.0, dim2=None, dim3=None)
#calibrate_crop(test_input, test_output, balance=0.0, dim2=None, dim3=None)



