%% Code for analysing active trajectories
% and save it to a txt file
clc
clear all

% Folder path for the data, change it to your folder
inputDataFolder = 'F:\Thomas_Suphona\master_thesis\version3\new_converted_data';
outputDataFolder = 'F:\Thomas_Suphona\master_thesis\version3';
D = dir(inputDataFolder);
nbrData = length(D);

fid = fopen('active_filter.txt', 'wt');
fprintf(fid, 'shows the number of active trajectories after some filter\n');
fprintf(fid, 'experiment, True, Tracked, OneGroup, FramesThreshold, Remains\n');

for iData = 3:nbrData
    filePath = fullfile(inputDataFolder, D(iData).name);%'2W300C20B.mat');
    [~,nameExperiment,~] = fileparts(filePath);
    data = load(filePath);
    fprintf(fid, '%s, ', nameExperiment);
    
    nbrVideoFrames = data.Data.nFrames;
    nbrActive_true = data.Data.NA_true;
    nbrActive_tracked = data.Data.NA_tracked;
    px2m_Active = data.Data.pixels2meterA;
    fr2sec = 1/data.Data.FrameRate_Hz;
    fprintf(fid, '%d, %d, ', nbrActive_true, nbrActive_tracked);
     
    fprintf('----------------------------------\n')
    fprintf('%d/%d files processed\n', iData, nbrData);
    fprintf('Processing active trajectories for %s\n', nameExperiment);
    fprintf('nbr frames: %d\n', nbrVideoFrames);
    fprintf('active true: %d\n', nbrActive_true);
    fprintf('active tracked: %d\n', nbrActive_tracked);
    fprintf('----------------------------------\n')
    
  
    activeTraj_x = full(data.Data.AtrajectoriesX_px)*px2m_Active;
    activeTraj_y = full(data.Data.AtrajectoriesY_px)*px2m_Active;
    
    nbrFramesActive = size(activeTraj_x, 2);
    boolActive = zeros(nbrActive_tracked, 1);
    
    % First filter using only trajs with one group
    for iActiveParticle = 1:nbrActive_tracked
         currActiveParticleX = activeTraj_x(iActiveParticle, :);
         propsActiveX = regionprops(currActiveParticleX~=0, currActiveParticleX, 'PixelValues');
         numGroupsActive = length(propsActiveX);
         if numGroupsActive == 1
            boolActive(iActiveParticle, 1) = 1; 
         end
         
    end
   
    nbrActiveParticlesLeft = sum(boolActive == 1);
    idxKeepActive = find(boolActive);
    
    fprintf(fid, '%d, ', nbrActiveParticlesLeft);
    
    % Second filter using frames threshold
    startFrames = 100;
    stepFrames = 100;
   
    while abs(nbrActive_true - nbrActiveParticlesLeft) > 5
        framesCount = sum(activeTraj_x(idxKeepActive, :)~=0, 2);
        idxKeepActive = idxKeepActive(framesCount > startFrames);
        nbrActiveParticlesLeft = length(idxKeepActive);
        startFrames = startFrames + stepFrames;
    end
    
    fprintf(fid, '%d, %d\n', startFrames, nbrActiveParticlesLeft);

end

fclose(fid);