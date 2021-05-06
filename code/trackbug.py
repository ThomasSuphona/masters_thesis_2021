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


def prepare_files():
    videosPath = 'D:/Thomas_Suphona/academia/master_thesis/version3/gray_scale_videos/*'
    listOfFiles = glob.glob(videosPath)
    listOfFiles = natsorted(listOfFiles)

    for ifile, file in enumerate(listOfFiles):
        fileName = ntpath.basename(file)
        nameExperiment = fileName.split('.')[0]
        fileExtension = fileName.split('.')[1]
        newpath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/code/image_sequences/'+nameExperiment
        if not os.path.exists(newpath):
            os.makedirs(newpath)

def convert_to_imagesequences():
    videosPath = 'D:/Thomas_Suphona/academia/master_thesis/version3/gray_scale_videos/0W900C20B*'
    listOfFiles = glob.glob(videosPath)
    listOfFiles = natsorted(listOfFiles)

    for ifile, file in enumerate(listOfFiles[:]):
        fileName = ntpath.basename(file)
        nameExperiment = fileName.split('.')[0]
        fileExtension = fileName.split('.')[1]
        outputPath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/code/' \
                     'image_sequences/'+nameExperiment+'/frame-%04d.png'

        print(file)
        fh = av.open(file)
        video = fh.streams.video[0]
        total_frames = video.frames
        frame_rate = video.framerate
        height = video.height
        width = video.width
        extension = fh.format.name
        duration_sec = float(video.duration * video.time_base)

        for frame in fh.decode(video=0):
            frame.to_image().save(outputPath % frame.index)

def read_imagesequences(imagePath):
    @pims.pipeline
    def gray(image):
        return image[:, :, 1]  # Take just the green channel

    listOfFolders = glob.glob(imagePath)
    listOfFolders = natsorted(listOfFolders)

    for ifolder, folder in enumerate(listOfFolders[:]):
        folderName = ntpath.basename(folder)
        print('Reading image sequences: '+folderName)
        imagePath = folder+'/*.png'

    frames = gray(pims.ImageSequence(imagePath))


    return frames

def locate_bug_one_frame(frames):
    n = 2

    diameter = 17
    separation = 20
    minmass = 5000 # 3450
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
    minmass = 3450
    nbrFrames = 1000
    maxDisp = 20
    mem = 15
    #mpp = 701.37
    #fps = 30

    imagePath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/code/image_sequences/0W900C20B*'
    outputPath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/code/traj_data/'
    listOfFolders = glob.glob(imagePath)
    listOfFolders = natsorted(listOfFolders)

    for ifolder, folder in enumerate(listOfFolders[:]):
        folderName = ntpath.basename(folder)
        imagePath = folder + '/*.png'
        outputPathData = outputPath+folderName+'.h5'
        print(folderName)


        frames = gray(pims.ImageSequence(imagePath))
        nbrFramesTot = len(frames)

        with tp.PandasHDFStore(outputPathData) as s:
            tp.quiet()
            tp.batch(frames[:], diameter=diameter, invert=False,
                     minmass=minmass, separation=separation, output=s)

            for linked in tp.link_df_iter(s, maxDisp, memory=mem):
                s.put(linked)
        s.close()


#imagePath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/code/image_sequences/0W900C1B*'
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
