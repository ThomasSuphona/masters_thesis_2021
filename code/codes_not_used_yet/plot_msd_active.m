%% plot msd for active particles

clc
clear all

% Folder path for the data, change it to your folder
inputDataFolder = 'F:\Thomas_Suphona\master_thesis\version3\msd_data';
D = dir(inputDataFolder);
nbrData = length(D);

for iData = 3:3%nbrData
    %filePath = fullfile(inputDataFolder, D(iData).name);
    filePath = 'F:\Thomas_Suphona\master_thesis\version3\msd_data\3W500C2B_msd.mat';
    data = load(filePath);
    nameExperiment = data.Data.name;
    msdActive = data.Data.MSDActive;
    nbrFrames = length(msdActive);
    tau = [0:1:nbrFrames-1];
    
    hold on
    loglog(tau, msdActive, 'DisplayName', nameExperiment);
end

hold off
set(gca, 'XScale', 'log', 'YScale', 'log');
legend('Location','southeast')
legend show
title('Mean square displacement for Active particles')
xlabel('\tau [s]')
ylabel('msd(\tau) [m^2]')