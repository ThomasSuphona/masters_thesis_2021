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

def video_to_trajectories():


    @pims.pipeline
    def gray(image):
        return image[:, :, 1]  # Take just the green channel


    videosPath = 'D:/Thomas_Suphona/academia/master_thesis/version3/gray_scale_videos/*'
    listOfFiles = glob.glob(videosPath)
    listOfFiles = natsorted(listOfFiles)

    for ifile, file in enumerate(listOfFiles[:]):
        fileName = ntpath.basename(file)
        nameExperiment = fileName.split('.')[0]
        fileExtension = fileName.split('.')[1]
        outputPath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/temp/frame-%04d.png'

        
        fh = av.open(file)
        video = fh.streams.video[0]
        total_frames = video.frames
        frame_rate = video.framerate
        height = video.height
        width = video.width
        extension = fh.format.name
        duration_sec = float(video.duration * video.time_base)

        print('coverting '+nameExperiment+'video to image sequences\n')
        for frame in fh.decode(video=0):
            frame.to_image().save(outputPath % frame.index)

        imagePath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/temp/*.png'
        frames = gray(pims.ImageSequence(imagePath))
        nbrFramesTot = len(frames)
        print('Reading '+nameExperiment+' image sequences: %d frames\n'%nbrFramesTot)

        diameter = 17
        separation = 20
        minmass = 3450
        nbrFrames = 1000
        maxDisp = 20
        mem = 15

        outputPathData = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/traj_data_new/'\
            +nameExperiment+'.h5'

        print('Tracking '+nameExperiment)
        with tp.PandasHDFStore(outputPathData) as s:
            tp.quiet()
            tp.batch(frames[:], diameter=diameter, invert=False,
                     minmass=minmass, separation=separation, output=s)

            for linked in tp.link_df_iter(s, maxDisp, memory=mem):
                s.put(linked)
        s.close()

        print('Remove images')
        for f in glob.glob('C:/Users/THOMAS/Desktop/masters_thesis_2021/temp/*.png'):
            os.remove(f)
        print('Done!!')

video_to_trajectories()