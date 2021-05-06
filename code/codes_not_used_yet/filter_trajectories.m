%% Main
% filtering bad particles step 1

clc
clear all

% Folder path for the data, change it to your folder
dataFolder = 'F:\Thomas_Suphona\master_thesis\version3\new_converted_data';
D = dir(dataFolder);
nbrData = length(D);

fid = fopen('filter_1.txt', 'wt');
fprintf(fid, 'experiment,PassiveBefore,PassiveAfter,ActiveBefore,ActiveAfter\n');

for iData = 3:nbrData
    filePath = fullfile(dataFolder, D(iData).name);
    [~,name,~] = fileparts(filePath);
    data = load(filePath);
    
    nbrFrames = data.Data.nFrames;

    np_true = data.Data.NP_true;
    np_tracked = data.Data.NP_tracked;

    na_true = data.Data.NA_true;
    na_tracked = data.Data.NA_tracked;
    
    fprintf('------------------------------------------\n');
    fprintf('%s\n', name);
    fprintf('Before filter\n');
    fprintf('nbr frames: %d\n', nbrFrames);

    fprintf('nbr active, true: %d\n', na_true);
    fprintf('nbr active, tracked: %d\n', na_tracked);

    fprintf('nbr passive, true: %d\n', np_true);
    fprintf('nbr passive, tracked: %d\n', np_tracked);

    fprintf('After filter\n')

    activeTrajX = data.Data.AtrajectoriesX_px;
    [activeRowIdx,~,~] = find(activeTrajX);
    [activeNNZFrames, ~] = histcounts(activeRowIdx);
    activeRowIdxKeep = find(activeNNZFrames > ceil(nbrFrames/2));

    fprintf('nbr active, true: %d\n', na_true);
    fprintf('nbr active, tracked: %d\n', length(activeRowIdxKeep));



    passiveTrajX = data.Data.PtrajectoriesX_px;
    [passiveRowIdx,~,~] = find(passiveTrajX);
    [passiveNNZFrames, ~] = histcounts(passiveRowIdx);
    passiveRowIdxKeep = find(passiveNNZFrames > ceil(nbrFrames/2));

    fprintf('nbr passive, true: %d\n', np_true);
    fprintf('nbr passive, tracked: %d\n', length(passiveRowIdxKeep));
    fprintf('------------------------------------------\n');
    
    
    fprintf(fid, '%s,%d,%d,%d,%d\n', name, np_tracked, ...
        length(passiveRowIdxKeep), na_tracked, ...
        length(activeRowIdxKeep));
    
end

fclose(fid);
