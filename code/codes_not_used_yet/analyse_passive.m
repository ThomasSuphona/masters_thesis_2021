%% Code for analysing passive trajectories
% and save it to a txt file
clc
clear all

% Folder path for the data, change it to your folder
inputDataFolder = 'F:\Thomas_Suphona\master_thesis\version3\new_converted_data';
outputDataFolder = 'F:\Thomas_Suphona\master_thesis\version3';
D = dir(inputDataFolder);
nbrData = length(D);

fid = fopen('passive_filter.txt', 'wt');
fprintf(fid, 'shows the number of passive trajectories after some filter\n');
fprintf(fid, 'experiment, True, Tracked, OneGroup, FramesThreshold, Remains\n');

for iData = 3:nbrData
    filePath = fullfile(inputDataFolder, D(iData).name);%'2W300C20B.mat');
    [~,nameExperiment,~] = fileparts(filePath);
    data = load(filePath);
    fprintf(fid, '%s, ', nameExperiment);
    
    nbrVideoFrames = data.Data.nFrames;
    nbrPassive_true = data.Data.NP_true;
    nbrPassive_tracked = data.Data.NP_tracked;
    px2m_Passive = data.Data.pixels2meterP;
    fr2sec = 1/data.Data.FrameRate_Hz;
    fprintf(fid, '%d, %d, ', nbrPassive_true, nbrPassive_tracked);
     
    fprintf('----------------------------------\n')
    fprintf('%d/%d files processed\n', iData, nbrData);
    fprintf('Processing passive trajectories for %s\n', nameExperiment);
    fprintf('nbr frames: %d\n', nbrVideoFrames);
    fprintf('passive true: %d\n', nbrPassive_true);
    fprintf('passive tracked: %d\n', nbrPassive_tracked);
    fprintf('----------------------------------\n')
    
  
    passiveTraj_x = full(data.Data.PtrajectoriesX_px)*px2m_Passive;
    passiveTraj_y = full(data.Data.PtrajectoriesY_px)*px2m_Passive;
    
    nbrFramesPassive = size(passiveTraj_x, 2);
    boolPassive = zeros(nbrPassive_tracked, 1);
    
    % First filter using only trajs with one group
    for iPassiveParticle = 1:nbrPassive_tracked
         currPassiveParticleX = passiveTraj_x(iPassiveParticle, :);
         propsPassiveX = regionprops(currPassiveParticleX~=0, currPassiveParticleX, 'PixelValues');
         numGroupsPassive = length(propsPassiveX);
         if numGroupsPassive == 1
            boolPassive(iPassiveParticle, 1) = 1; 
         end
         
    end
   
    nbrPassiveParticlesLeft = sum(boolPassive == 1);
    idxKeepPassive = find(boolPassive);
    
    fprintf(fid, '%d, ', nbrPassiveParticlesLeft);
    
    % Second filter using frames threshold
    startFrames = 100;
    stepFrames = 100;
   
    while abs(nbrPassive_true - nbrPassiveParticlesLeft) > 60
        framesCount = sum(passiveTraj_x(idxKeepPassive, :)~=0, 2);
        idxKeepPassive = idxKeepPassive(framesCount > startFrames);
        nbrPassiveParticlesLeft = length(idxKeepPassive);
        startFrames = startFrames + stepFrames;
    end
    
    fprintf(fid, '%d, %d\n', startFrames, nbrPassiveParticlesLeft);

end

fclose(fid);