%% Unit test

clc
clear all

folderName = 'Positions/New_Conditions_2/0W0C16B';
fileName = 'position.mat';
file = fullfile(folderName, fileName);

pos = load(file);

nbrTaus = 100;
tau = linspace(1, nbrTaus);

msd = msd(pos, 0, tau);

size(msd)