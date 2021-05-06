clear all; close all; clc;
% edit('example_dvm2dthreshold')

% Load video file

filepath='F:\UndistortedVideos_2\';
listing=dir([filepath '*.avi']);
%addpath('F:\UndistortedVideos\')
for l = 1:length(listing)  

FileName=listing(l).name

%[FileName,FilePath]=uigetfile('*.avi');
%X.FileName=FileName;
%X.Filepath=FilePath;
    
% fileName = strcat(names{ii},'.avi');
%filePath = 'F:\UndistortedVideos';
    
video = VideoReader([filepath FileName]);
%video = VideoFileAvi('FileName',FileName)%,'FilePath',filePath)

% % Create DVM
% dvm = DVM2DThreshold(video);
% 
% % Video properties
% filetype = dvm.filetype()
% framenumber = dvm.framenumber()
% framerate = dvm.framerate()
% 
% % Read video portion

%Create the output video
v = VideoWriter(fullfile('F:\','BlackAndWhiteVideos_2',FileName));
writerObj.FrameRate = 30;
open(v);

while hasFrame(video)
     %read the frame from the input video
     img = readFrame(video); 
    
    %make the image gray
    grayImage = rgb2gray(img);
    grayFrame = cat(3, grayImage, grayImage, grayImage);
    BB=imcomplement(grayFrame);
    
    %write the image into the output video
    writeVideo(v,BB);
end
close(v);

end