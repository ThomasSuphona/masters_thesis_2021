%% main converting data to a structure similar to giorgios 
% Might need to do some filtering to delete short trajectories in 
% Before using this make sure that there are 364 cylinder data
% 364 bugs data and 364 cut videos. If you have more, move the 
% the videos and data that corresponds to experiments with 0 cylinders
% somewhere else i.e the experiments with name 0C in it. 

clc
clear all

tic

otsFolderPath = 'E:\Thomas_Suphona\master_thesis\version3\code\code_version2\ots1.0.1';
otsFileName = 'OTS.m';
otsPath = fullfile(otsFolderPath, otsFileName);
run(otsPath)

% Here change folder to where you keep the data
passiveFolderPath = 'E:\Thomas_Suphona\master_thesis\version3\no_obstacles_data\unconverted\cylinder_data';
activeFolderPath = 'E:\Thomas_Suphona\master_thesis\version3\no_obstacles_data\unconverted\bugs_data';
videosFolderPath = 'E:\Thomas_Suphona\master_thesis\version3\no_obstacles_data\videos';
saveFolderPath = 'E:\Thomas_Suphona\master_thesis\version3\no_obstacles_data\converted';

activeDir = dir(activeFolderPath);
nExperiments = length(activeDir);

passivePathExtension = '.avi_DVM.mat';
activePathExtension = 'position8.mat';
videoPathExtension = '.mkv';

SizeActive = 21.5e-3; % [m] (radius of the active particles, bugs)
SizePassive = 9.5e-3; % [m] (radius of the passive particles, cylinders)
pixels2meter = 7.013730031e-4;

for iExperiment = 3:nExperiments
    
    nameExperiment = activeDir(iExperiment).name;
    
    activeFullPath = [activeFolderPath '\' nameExperiment '\' activePathExtension];
    passiveFullPath = [passiveFolderPath '\' nameExperiment passivePathExtension];
    videoFullPath = [videosFolderPath '\' nameExperiment videoPathExtension];
    
     Video = VideoReader(videoFullPath);
     FrameRate = Video.FrameRate;    % [fps]
     Duration = Video.Duration;      % [s]   
     nFrames = ceil(Duration*FrameRate);  % [frames] may not use this

    dataActive = load(activeFullPath);
    dataPassive = load(passiveFullPath);
    
    nbrTruePassive = dataActive.d.nCylinders;
    nbrTrueActive = dataActive.d.nBugs;
    
    nbrTrackedActive = length(dataActive.Trajs);
    nbrTrackedPassive = length(dataPassive.dvm.Trajectories);
    
    nbrFramesToTrackPassive = dataPassive.dvm.FramesToTrack;
    nbrFramesToTrackActive = dataActive.d.nImages;
    
    Data.VideoName = Video.Name;
    Data.FrameRate_Hz = FrameRate;
    Data.Duration_s = Duration;
    Data.nFrames = nFrames;
    Data.FrameHeight_px = Video.Height;
    Data.FrameWidth_px = Video.Width;
    Data.SizeActive_m = SizeActive;
    Data.SizePassive_m = SizePassive;
    Data.pixels2meterP = pixels2meter;
    Data.NP_true = nbrTruePassive;
    Data.NP_tracked = nbrTrackedPassive;
    
    
    PtrajectoriesX_px = spalloc(nbrTrackedPassive, nbrFramesToTrackPassive, ...
        nbrTrackedPassive*nbrFramesToTrackPassive);
    PtrajectoriesY_px = spalloc(nbrTrackedPassive, nbrFramesToTrackPassive, ...
        nbrTrackedPassive*nbrFramesToTrackPassive);
    
    
    AtrajectoriesX_px = spalloc(nbrTrackedActive, nbrFramesToTrackActive, ...
        nbrTrackedActive*nbrFramesToTrackActive);
    AtrajectoriesY_px = spalloc(nbrTrackedActive, nbrFramesToTrackActive, ...
        nbrTrackedActive*nbrFramesToTrackActive);
    
    for iPassive = 1:nbrTrackedPassive
       
        j_k = dataPassive.dvm.Trajectories(1, iPassive).T + 1;
        v_kx = dataPassive.dvm.Trajectories(1, iPassive).X;
        v_ky = dataPassive.dvm.Trajectories(1, iPassive).Y;
        
        % Maybe change here for speed and stack vector
        % and call sparse once'
        
        
        PtrajectoriesX_px(iPassive, j_k) = v_kx;
        PtrajectoriesY_px(iPassive, j_k) = v_ky;
    end
    
    
    for iActive = 1:nbrTrackedActive
       
        j_k = dataActive.Trajs(iActive).T;
        v_kx = dataActive.Trajs(iActive).X;
        v_ky = dataActive.Trajs(iActive).Y;
        
        % Maybe change here for speed and stack vector
        % and call sparse once'
        
        
        AtrajectoriesX_px(iActive, j_k) = v_kx;
        AtrajectoriesY_px(iActive, j_k) = v_ky;
    end
    
    Data.PtrajectoriesX_px = PtrajectoriesX_px;
    Data.PtrajectoriesY_px = PtrajectoriesY_px;
    
    Data.AtrajectoriesX_px = AtrajectoriesX_px;
    Data.AtrajectoriesY_px = AtrajectoriesY_px;

    Data.pixels2meterA = pixels2meter;
    Data.NA_true = nbrTrueActive;
    Data.NA_tracked = nbrTrackedActive;
    
    Data.NWeights_nbr =  dataActive.d.nWeigths;
    Data.NWeights_kg =  (dataActive.d.nWeigths*5 + 2)*0.001;
    
    
    saveFileName = [saveFolderPath '\' nameExperiment '.mat'];
    save(saveFileName, 'Data');
    
end

toc