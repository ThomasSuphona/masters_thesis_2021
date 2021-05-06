%The directory in which we read the the trajectories of the obstacles
%(which are in .mat files)
directory = '../TrackingCylinders/';
d = dir([directory '*.mat']);

%run OTS to be able to read the trajectories dvm.Trajectories
OTS;
%clear the command window
clc;
for iExperiment = 196:202%length(d) %15:20
%iExperiment = 110;

%The relation between pixels and mm
px2mm = 1/1.4;
%The relation between frame and second
fr2sec = 1/30;

%extract the trajectory in the .mat file
filePath = fullfile(directory, d(iExperiment).name);
traj = load(filePath);

iExperiment
d(iExperiment).name

%Find the number of Weigths
nbrTrueWeights = str2double( regexp( d(iExperiment).name, ...
    '\d+(?=W)', 'match' ));

%Find the number of Cylinders
nbrTrueCylinders = str2double( regexp( d(iExperiment).name, ...
    '(?<=W)\d+(?=C)', 'match' ));

%Find the number of Bugs
nbrTrueBugs = str2double( regexp( d(iExperiment).name, ...
    '(?<=C)\d+(?=B)', 'match' ));

%number of frame of the video
nImages = traj.dvm.FramesToTrack;
%extract the trajectory
trajs = traj.dvm.Trajectories;

%Maximal length of the trajectories we want to keep
lengthMax = floor(nImages/2); 

%Delete the trajectories that are too short 
IdxDeleteTraj = [];
for ii=1:length(trajs)
    if length(trajs(ii).T) < lengthMax
        IdxDeleteTraj = [IdxDeleteTraj ii];
    end
end
IdxDeleteTraj = sort(IdxDeleteTraj,'descend');
trajs(IdxDeleteTraj) = [];

%Delete the trajectories with a time gap in them
IdxDeleteTraj = [];
for ii = 1:length(trajs)
    if (trajs(ii).T(end) - trajs(ii).T(1) + 1) ~= length(trajs(ii).T)
        IdxDeleteTraj = [IdxDeleteTraj ii];
    end    
end
trajs(IdxDeleteTraj) = [];

%number of trajectories we have kept
nbrCylinders = length(trajs);

%L(i) will be the length of the i-th trajectory
L=[];
msdsum = zeros(nImages-1, 1);

for iCylinder = 1:nbrCylinders

    posX = trajs(iCylinder).X;
    posY = trajs(iCylinder).Y;
    L(iCylinder) = length(posX);

    msdsum = msdsum + msd(posX*px2mm, posY*px2mm,nImages)*L(iCylinder);
end

for ii=1:nImages-1
    msdsum(ii,1) = msdsum(ii,1)/(sum(L(L>=ii)));
    %msdsum = msdsum/(sum(L));
end
%Delete the values that are 0 in the msd (it happens if all the
%trajectories are shorter than the number of images)
msdsum = msdsum(msdsum~=0);

%Show the number of cylinders we kept and the real number of cylinders
fprintf('%d/%d cylinders processed\n', nbrCylinders, nbrTrueCylinders);

%Save the data
save(sprintf('%dW%dC%dB',nbrTrueWeights,nbrTrueCylinders,nbrTrueBugs),...
    'msdsum');
movefile([sprintf('%dW%dC%dB',nbrTrueWeights,nbrTrueCylinders,nbrTrueBugs) '.mat'], ...
    '.\MsdCylinders_1');

%plot
tau = linspace(1, length(msdsum), length(msdsum));
plot(tau.*fr2sec, msdsum, 'DisplayName', ...
    sprintf('%dW%dC%dB', nbrTrueWeights, nbrTrueCylinders, ...
    nbrTrueBugs));
hold on 
end

%% Plot

   

hold off
set(gca, 'XScale', 'linear', 'YScale', 'linear');
legend('Location','southeast')
legend show
title('Mean square displacement')
xlabel('\tau [s]')
ylabel('msd(\tau) [mm^2]')
