%% plot msd for passive particles

clc
clear all

% Folder path for the data, change it to your folder
inputDataFolder = 'F:\Thomas_Suphona\master_thesis\version3\msd_data';
D = dir(inputDataFolder);
nbrData = length(D);

for iData = 3:nbrData
    filePath = fullfile(inputDataFolder, D(iData).name);
    data = load(filePath);
    nameExperiment = data.Data.name;
    msdPassive = data.Data.MSDPassive;
    nbrFrames = length(msdPassive);
    tau = [0:1:nbrFrames-1];
    
    hold on
    loglog(tau, msdPassive, 'DisplayName', nameExperiment);
end

hold off
set(gca, 'XScale', 'log', 'YScale', 'log');
legend('Location','southeast')
legend show
title('Mean square displacement for Passive particles')
xlabel('\tau [s]')
ylabel('msd(\tau) [m^2]')