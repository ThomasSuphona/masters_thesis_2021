% Series of examples to demonstrate the use of DVM2DThreshold.
% 
% See also DVM2DThreshold.
% 
%   Author: Giuseppe Pesce, Giovanni Volpe
%   Revision: 1.0.0
%   Date: 2015/01/01

clear all; close all; clc;
% edit('example_dvm2dthreshold')

addpath('.\ots1.0.1\dvm');

% Load video file
filepath='.\BlackAndWhiteVideos\';
listing=dir([filepath '*.avi']);

for l=1:length(listing)
tic 
FileName=listing(l).name;
%video = VideoReader([filepath FileName]);
video = VideoFileAvi('FileName',FileName,'Filepath',filepath)

% Create DVM
dvm = DVM2DThreshold(video);

% Video properties
filetype = dvm.filetype()
framenumber = dvm.framenumber()
framerate = dvm.framerate()

% Read video portion


% v = VideoWriter(num2str(video.FileName));
% writerObj.FrameRate = framerate;
% open(v);
% 
% for i=1:10
%      
%  images = dvm.read(i,i);
% %   figure
% %   image(images)
% 
% grayImage = rgb2gray(images);
% grayFrame = cat(3, grayImage, grayImage, grayImage);
% BB=imcomplement(grayFrame);
% 
% 
% % figure(3)
% % image(BB)
% 
% 
% 
% writeVideo(v,BB);
% end
% close(v);



% Play video
%dvm.play()

% Tracking
dvm = dvm.tracking('verbose', false, ...
    'displayon', false, ...
    'FramesToTrack', framenumber, ...
    'VideoPortion', 100, ...
    'MinParticleRadius', 2, ...
    'MaxParticleRadius', 4.5, ...
    'PositiveMask', true, ...
    'Threshold', 220, ...
    'ErodeRadius', 150, ...
    'DilateRadius', 100);

% Tracing
dvm = dvm.tracing('verbose', false, ...
    'displayon', false, ...
    'MaxDistance', 25, ...
    'MaxHiatus', 15);

% minTraj = 10;
% deleteIndex = [];
% %delete the trajectories that are less than minTraj
% for ii=1:length(dvm.Trajectories)
%     if length(dvm.Trajectories(ii).T) <100
%         disp(ii)
%         disp(length(dvm.Trajectories(ii).T))
%     end
%     if length(dvm.Trajectories(ii).T) < minTraj
%         deleteIndex = [deleteIndex,ii]; 
%     end
% end
% dvm.Trajectories(deleteIndex) = [];

save(['.\TrackingCylinders\' num2str(video.FileName) '_DVM.mat'],'dvm');
toc

end