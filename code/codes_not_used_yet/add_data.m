%% Calculate MSD and save to file

clc 
clear all

trajInputFolder = 'F:\Thomas_Suphona\master_thesis\version3\new_converted_data';
indicesInputFolder = 'F:\Thomas_Suphona\master_thesis\version3\indices_data';
orientationInputFolder = 'F:\Thomas_Suphona\master_thesis\version2\code\bugs_data';
outputFolder = 'F:\Thomas_Suphona\master_thesis\version3\main_data';

D = dir(trajInputFolder);
nbrData = length(D);

for iData = 3:nbrData
    trajFilePath = fullfile(trajInputFolder, D(iData).name);
    [~,nameExperiment,~] = fileparts(trajFilePath);
    
    indicesFilePath = [indicesInputFolder '\' nameExperiment '_indices.mat'];
    orientationFilePath = [orientationInputFolder '\' nameExperiment '\' 'position8.mat'];
    
    load(trajFilePath);
    dataIndices = load(indicesFilePath);
    dataOrientation = load(orientationFilePath);
    
    nbrTrackedActive = Data.NA_tracked;
    nbrFramesToTrackActive = dataOrientation.d.nImages;
    
    AOrientation = spalloc(nbrTrackedActive, nbrFramesToTrackActive, ...
       nbrTrackedActive*nbrFramesToTrackActive);
    
   
   for iActive = 1:nbrTrackedActive
       
        j_k = dataOrientation.Trajs(iActive).T;
        v_kx = dataOrientation.Trajs(iActive).Orientation;
        
        % Maybe change here for speed and stack vector
        % and call sparse once'
        
        
        AOrientation(iActive, j_k) = v_kx;
  
    end
    
    Data.PIndices = dataIndices.Data.PassiveIndices;
    Data.AIndices = dataIndices.Data.ActiveIndices;
    Data.AOrientation = AOrientation;
    
    fprintf('----------------------------------\n')
    fprintf('%d/%d files processed\n', iData, nbrData);
    fprintf('Processing data for %s\n', nameExperiment);
    fprintf('----------------------------------\n')
    
    saveFileName = [outputFolder '\' nameExperiment '.mat'];
    save(saveFileName, 'Data');
end