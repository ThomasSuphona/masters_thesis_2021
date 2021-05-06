%% Calculate the index of the particles that has 
% reasonable number of frames that can be used

clc
clear all

% Folder path for the data, change it to your folder
inputDataFolder = 'F:\Thomas_Suphona\master_thesis\version3\new_converted_data';
outputDataFolder = 'F:\Thomas_Suphona\master_thesis\version3\indices_data';
D = dir(inputDataFolder);
nbrData = length(D);


for iData = 3:nbrData
    
    filePath = fullfile(inputDataFolder, D(iData).name);
    [~,nameExperiment,~] = fileparts(filePath);
    data = load(filePath);
    
    % General data
    nbrVideoFrames = data.Data.nFrames;
    fr2sec = 1/data.Data.FrameRate_Hz;
    
    % Passive particles data
    nbrPassive_true = data.Data.NP_true;
    nbrPassive_tracked = data.Data.NP_tracked;
    px2m_Passive = data.Data.pixels2meterP;
    passiveTraj_x = full(data.Data.PtrajectoriesX_px)*px2m_Passive;
    passiveTraj_y = full(data.Data.PtrajectoriesY_px)*px2m_Passive;
    nbrFramesPassive = size(passiveTraj_x, 2);
    boolPassive = zeros(nbrPassive_tracked, 1);
    
    % Active particles data
    nbrActive_true = data.Data.NA_true;
    nbrActive_tracked = data.Data.NA_tracked;
    px2m_Active = data.Data.pixels2meterA;
    activeTraj_x = full(data.Data.AtrajectoriesX_px)*px2m_Active;
    activeTraj_y = full(data.Data.AtrajectoriesY_px)*px2m_Active;
    nbrFramesActive = size(activeTraj_x, 2);
    boolActive = zeros(nbrActive_tracked, 1);
    
    % Print stuff on screen
    fprintf('----------------------------------\n')
    fprintf('%d/%d files processed\n', iData, nbrData);
    fprintf('Processing trajectories for %s\n', nameExperiment);
    fprintf('nbr frames: %d\n', nbrVideoFrames);
    fprintf('----------------------------------\n')
    
    
    % First passive filter using only trajs with one group
    %-------------------------------------------------------------
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
    
    % Second filter using frames threshold
    startFrames = 100;
    stepFrames = 100;
   
    while abs(nbrPassive_true - nbrPassiveParticlesLeft) > 60
        framesCountPassive = sum(passiveTraj_x(idxKeepPassive, :)~=0, 2);
        idxKeepPassive = idxKeepPassive(framesCountPassive > startFrames);
        nbrPassiveParticlesLeft = length(idxKeepPassive);
        startFrames = startFrames + stepFrames;
    end
    
    Data.PassiveIndices = idxKeepPassive;
    %-----------------------------------------------------------------------
    
    
    % First Active filter using only trajs with one group
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
    
    % Second filter using frames threshold
    startFrames = 100;
    stepFrames = 100;
   
    while abs(nbrActive_true - nbrActiveParticlesLeft) > 5
        framesCountActive = sum(activeTraj_x(idxKeepActive, :)~=0, 2);
        idxKeepActive = idxKeepActive(framesCountActive > startFrames);
        nbrActiveParticlesLeft = length(idxKeepActive);
        startFrames = startFrames + stepFrames;
    end
    
    Data.ActiveIndices = idxKeepActive;
    saveFileName = [outputDataFolder '\' nameExperiment '_indices.mat'];
    save(saveFileName, 'Data');
end