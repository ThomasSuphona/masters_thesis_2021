from __future__ import division, unicode_literals, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp
import av
import glob
import re
from natsort import natsorted
import ntpath
import os
import cv2
import time
from PIL import Image
import PIL.ImageOps as pimo


def prepare_files():
    videosPath = 'D:/Thomas_Suphona/academia/master_thesis/version1/videos/undistorted/C20B838P0M*'
    listOfFiles = glob.glob(videosPath)
    listOfFiles = natsorted(listOfFiles)

    for ifile, file in enumerate(listOfFiles):
        fileName = ntpath.basename(file)
        nameExperiment = fileName.split('.')[0]
        fileExtension = fileName.split('.')[1]
        newpath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/code/image_sequences_old/'+nameExperiment
        if not os.path.exists(newpath):
            os.makedirs(newpath)

def convert_to_imagesequences():
    videosPath = 'D:/Thomas_Suphona/academia/master_thesis/version1/videos/undistorted/C20B838P0M*'
    listOfFiles = glob.glob(videosPath)
    listOfFiles = natsorted(listOfFiles)

    for ifile, file in enumerate(listOfFiles[:]):
        fileName = ntpath.basename(file)
        nameExperiment = fileName.split('.')[0]
        fileExtension = fileName.split('.')[1]
        outputPath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/code/' \
                     'image_sequences_old/' + nameExperiment + '/frame-%04d.png'

        print(nameExperiment)
        fh = av.open(file)

        for frame in fh.decode(video=0):
            #if frame.index == 1:
            #    break
            frame.to_image().save(outputPath % frame.index)
            undistort_and_filter(outputPath % frame.index, outputPath % frame.index)

def undistort_and_filter(input_loc, output_loc):
    balance = 0.0
    dim2 = None
    dim3 = None

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
    cropped_img = grey_img[137:878, 530:1405]  # 435, 85, 440, 100
    # cv2.imshow("cropped", cropped_img)
    cv2.imwrite(output_loc, cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_imagesequences(imagePath):
    @pims.pipeline
    def gray(image):
        return image[:, :, 1]  # Take just the green channel

    listOfFolders = glob.glob(imagePath)
    listOfFolders = natsorted(listOfFolders)
    print(listOfFolders)

    for ifolder, folder in enumerate(listOfFolders[:]):
        folderName = ntpath.basename(folder)
        print('Reading image sequences: '+folderName)
        imagePath = folder+'/*.jpg'


    frames = pims.ImageSequence(imagePath)


    return frames

def locate_bug_one_frame(frames):
    n = 0

    diameter = 17
    separation = 20
    minmass = 3000 # 3450
    smoothing_size = None
    maxsize = None


    f = tp.locate(frames[n], diameter, minmass=minmass,
                  separation=separation, invert=False,
                  smoothing_size=smoothing_size,
                  maxsize=maxsize)

    fig, ax = plt.subplots()
    tp.annotate(f, frames[n])  #check features

    ax.hist(f['size'], bins=20)    # check mass
    # Optionally, label the axes.
    ax.set(xlabel='mass', ylabel='count')


    #f = tp.locate(frames[fr], 53, invert=False, minmass=40000) # try again
    #tp.annotate(f, frames[fr])

    #tp.subpx_bias(f)    # check subpixel accuracy

    #tp.subpx_bias(tp.locate(frames[0], 57, invert=False, minmass=40000)) # subpixel again
    plt.show()

def locate_multiple_frames(size, minmass, separation, frames, nbrFrames, topn):
    #tp.quiet()
    f = tp.batch(frames[:nbrFrames], size, minmass=minmass,
                 separation=separation, topn=topn, invert=False)
    return f

def link_traj(f, maxDisp, mem):
    #pred = tp.predict.NearestVelocityPredict()
    #t = pd.concat(pred.link_df_iter(f, search_range=maxDisp, memory=mem))
    t = tp.link(f, search_range=maxDisp, memory=mem)

    plt.figure()
    n = 0
    #tp.annotate(t, frames[n])
    print(t.head())

    t1 = tp.filter_stubs(t, 10)     # particle have to last for some frames
    # Compare the number of particles in the unfiltered and filtered data.
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())

    #tp.mass_size(t1.groupby('particle').mean())  # convenience function -- just plots size vs. mass

    #t2 = t1[(t1['mass'] > 75000)] #t1['mass'] > 40000 &
    t2 = t1



    #tp.annotate(t2[t2['frame'] == n], frames[n])

    tp.plot_traj(t2)# plot traj

    #d = tp.compute_drift(t2) # compute drift
    #d.plot()

    #tm = tp.subtract_drift(t2.copy(), d) #remove drift
    #ax = tp.plot_traj(tm)
    plt.show()

    return t2

def msd_individual(t, mpp, fps):
    im = tp.imsd(t, mpp, fps)

    fig, ax = plt.subplots()
    ax.plot(im.index, im, 'k-', alpha=0.1)  # black lines, semitransparent
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
           xlabel='lag time $t$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()

def locate_link_save_data():
    @pims.pipeline
    def gray(image):
        return image[:, :, 1]  # Take just the green channel

    diameter = 17
    separation = 20
    minmass = 3000
    nbrFrames = 1000
    maxDisp = 20
    mem = 15
    #mpp = 701.37
    #fps = 30

    imagePath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/image_sequences_old/20B491P2M'
    outputPath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/traj_data_old/'
    listOfFolders = glob.glob(imagePath)
    listOfFolders = natsorted(listOfFolders)

    for ifolder, folder in enumerate(listOfFolders[:]):
        folderName = ntpath.basename(folder)
        imagePath = folder + '/*.jpg'
        outputPathData = outputPath+folderName+'.h5'
        print(folderName)


        frames = pims.ImageSequence(imagePath)
        nbrFramesTot = len(frames)

        with tp.PandasHDFStore(outputPathData) as s:
            #tp.quiet()
            tp.batch(frames[:], diameter=diameter, invert=False,
                     minmass=minmass, separation=separation, output=s)

            for linked in tp.link_df_iter(s, maxDisp, memory=mem):
                s.put(linked)
        s.close()

def img_to_video():
    imagePath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/code/image_sequences_old/C20B838P0M'
    listOfFolders = glob.glob(imagePath)
    listOfFolders = natsorted(listOfFolders)

    for ifolder, folder in enumerate(listOfFolders[:]):
        folderName = ntpath.basename(folder)
        imagePath = folder + '/*.png'
        outputPath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/code/gray_scale_videos_old/' + \
                     folderName + '.avi'

        images = glob.glob(imagePath)
        images = natsorted(images)

        frame = cv2.imread(images[0])
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(outputPath, fourcc, 14.09, (width, height))

        for image in images:
            video.write(cv2.imread(image))

        cv2.destroyAllWindows()
        video.release()


#imagePath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/image_sequences_old/20B491P2M'
#frames = read_imagesequences(imagePath)
#totalNbrFrames = len(frames)
#print('Total number frames: ', totalNbrFrames)
#locate_bug_one_frame(frames)

#diameter = 17
#separation = 20
#minmass = 5000
#nbrFrames = 1000
#topn = None

#f = locate_multiple_frames(diameter, minmass, separation, frames, nbrFrames, topn)
#maxDisp = 20
#mem = 15
#t = link_traj(f, maxDisp, mem)
#msd_individual(t, mpp, fps)
#locate_link_save_data()
#convert_to_imagesequences()
#img_to_video()