%% Code for calculating MSD
% and save it to a file
clc
clear all

% Folder path for the data, change it to your folder
inputDataFolder = 'F:\Thomas_Suphona\master_thesis\version3\main_data_temp';
outputDataFolder = 'F:\Thomas_Suphona\master_thesis\version3\msd_data';
D = dir(inputDataFolder);
nbrData = length(D);

 
for iData = 3:nbrData
    filePath = fullfile(inputDataFolder, D(iData).name);
    [~,name,~] = fileparts(filePath);
    data = load(filePath);
    
    nbrFrames = data.Data.nFrames;
    fr2sec = 1/data.Data.FrameRate_Hz;
    
    np_true = data.Data.NP_true;
    np_tracked = data.Data.NP_tracked;
    np_use = length(data.Data.PIndices);
    pIndices = data.Data.PIndices;
    px2m_p = data.Data.pixels2meterP;
    
    na_true = data.Data.NA_true;
    na_tracked = data.Data.NA_tracked;
    na_use = length(data.Data.AIndices);
    aIndices = data.Data.AIndices;
    px2m_a = data.Data.pixels2meterA;
        
     
    fprintf('----------------------------------\n')
    fprintf('calculating MSD for %s\n', name);
    fprintf('nbr frames: %d\n', nbrFrames);
    fprintf('active true: %d\n', na_true);
    fprintf('active use: %d\n', na_use);
    fprintf('passive true: %d\n', np_true);
    fprintf('passive use: %d\n', np_use);
    fprintf('----------------------------------\n')
     
    pTraj_x = data.Data.PtrajectoriesX_px(pIndices, :)*px2m_p;
    pTraj_y = data.Data.PtrajectoriesY_px(pIndices, :)*px2m_p;
    
    aTraj_x = data.Data.AtrajectoriesX_px(aIndices, :)*px2m_a;
    aTraj_y = data.Data.AtrajectoriesY_px(aIndices, :)*px2m_a;
    
    
    nbrFramesP = size(pTraj_x, 2);
    nbrFramesA = size(aTraj_x, 2);
    
    msdSumP = zeros(nbrFramesP, 1);
    msdSumA = zeros(nbrFramesA, 1);
    
    % maybe use fraction for particles with lower number of frames
    for iAParticle = 1:na_use
        currATraj_x = nonzeros(aTraj_x(iAParticle, :));
        currATraj_y = nonzeros(aTraj_y(iAParticle, :));
        
        msdrA = msd(currATraj_x, currATraj_y);
        msdSumA = msdSumA + [msdrA;zeros(length(msdSumA)-length(msdrA), 1)];
         
    end
    
    for iPParticle = 1:np_use
        currPTraj_x = nonzeros(pTraj_x(iPParticle, :));
        currPTraj_y = nonzeros(pTraj_y(iPParticle, :));
        
        msdrP = msd(currPTraj_x, currPTraj_y);
        msdSumP = msdSumP + [msdrP;zeros(length(msdSumP)-length(msdrP), 1)];
         
    end
   
    Data.MSDActive = msdSumA;
    Data.MSDPassive = msdSumP;
    Data.name = name;
    saveFileName = [outputDataFolder '\' name '_msd.mat'];
    save(saveFileName, 'Data');
end


