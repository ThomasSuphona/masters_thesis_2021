% s(i).isTraj(j) = -1 if the point belongs to a trajectory
%                   0 if it doesn't belong to a trajectory
%                   j if it is the first point of the the j-th trajectory
%
% frame(i).indexTraj is a vector with all the indices of the trajectories
%which end at the frame number i
%
%Trajs is a struct array where Trajs(j) corresponds to the trajectory 
%number j. Trajs has the fieldnames
% Area
% X
% Y
% Orientation
% MajorAxisLength
% MinorAxisLength
% Eccentricity
% T 
% P : Trajs(i).P(j) is the position in the array s(Trajs(i).T(j)).fieldname
%     of the corresponding point
clear all, close all

listing = dir('.\Elisa0W0C1B');

for ii=3:length(listing)
ii
%The folder in which we read the images
nameFolder = listing(ii).name

%load the file position.mat given by detectBugs_3
name = fullfile('.',nameFolder,'position.mat');
load(name)

%Folders of the binarized image sequences
parentFolder = fullfile('..\..\Image_Sequences\Binarized');
folder = fullfile(parentFolder,nameFolder);
%Images folder
D=dir([folder '\*.tiff']);

%Add all the useful functions to the path
addpath('.\util')


%% Initialisation

%Create the first trajectories for the first frame
[s,Trajs,frame] = initTrajs(s,d,700,1300);

%%
disp('phase 1')
[s,d,Trajs,frame] = detectTrajsDist(s,d,Trajs,frame,[14 20],700,1300);

save(fullfile('.',nameFolder,'position2.mat'),'s','d','Trajs','frame')

%%
disp('phase 2')
load(fullfile('.',nameFolder,'position2.mat'))

for dist=[20 30 40 50 60]
    for k=1:100
        [s,d,Trajs,frame] = detectTrajArea2(s,d,Trajs,frame,dist,1800,2400);
        [s,d,Trajs,frame] = detectTrajArea3(s,d,Trajs,frame,dist,3000,3500);
        [s,d,Trajs,frame] = detectTrajArea4(s,d,Trajs,frame,dist,4000,4500);
    end
    [s,d,Trajs,frame] = detectTrajsDist(s,d,Trajs,frame,[14 20 dist],700,1300);

end
save(fullfile('.',nameFolder,'position3.mat'),'s','d','Trajs','frame')
end

%%
disp('phase 3')
load(fullfile('.',nameFolder,'position3.mat'))

[s,d,Trajs,frame] = detectTrajsDist(s,d,Trajs,frame,[14 20],600,1400);

for dist=[20 30 40 50 60]
    for k=1:100
        [s,d,Trajs,frame] = detectTrajArea2(s,d,Trajs,frame,dist,1600,2600);
        [s,d,Trajs,frame] = detectTrajArea3(s,d,Trajs,frame,dist,2800,3700);
        [s,d,Trajs,frame] = detectTrajArea4(s,d,Trajs,frame,dist,3800,4800);
        [s,d,Trajs,frame] = detectTrajArea5(s,d,Trajs,frame,dist,4800,5200);
        [s,d,Trajs,frame] = detectTrajArea6(s,d,Trajs,frame,dist,5200,6000);
    end
    [s,d,Trajs,frame] = detectTrajsDist(s,d,Trajs,frame,[14 20 dist],600,1400);
end

save(fullfile('.',nameFolder,'position4.mat'),'s','d','Trajs','frame')

%%
disp('phase 4')
load(fullfile('.',nameFolder,'position4.mat'))
[s,d,Trajs,frame] = detectTrajsDist(s,d,Trajs,frame,[14 20],500,1500);

for dist=[20 30 40 50 60 70 80 100]
    for k=1:100
        [s,d,Trajs,frame] = detectTrajArea2(s,d,Trajs,frame,dist,1500,2800);
        [s,d,Trajs,frame] = detectTrajArea3(s,d,Trajs,frame,dist,2600,4000);
        [s,d,Trajs,frame] = detectTrajArea4(s,d,Trajs,frame,dist,3500,5200);
        [s,d,Trajs,frame] = detectTrajArea5(s,d,Trajs,frame,dist,4600,5400);
        [s,d,Trajs,frame] = detectTrajArea6(s,d,Trajs,frame,dist,5000,6200);
        [s,d,Trajs,frame] = detectTrajArea7(s,d,Trajs,frame,dist,6200,1e4);
    end
    [s,d,Trajs,frame] = detectTrajsDist(s,d,Trajs,frame,[14 20 dist],500,1500);
end

save(fullfile('.',nameFolder,'position5.mat'),'s','d','Trajs','frame')
%end
%% Detect Areas cutted in several parts
disp('phase 5')
load(fullfile('.',nameFolder,'position5.mat'))

[s,d,Trajs,frame] = detectTrajAreaHalf(s,d,Trajs,frame,60,100,500);
[s,d,Trajs,frame] = detectTrajsDist(s,d,Trajs,frame,[14 20 40 60],500,1500);
[s,d,Trajs,frame] = detectTrajsDist(s,d,Trajs,frame,60,400,1500);

save(fullfile('.',nameFolder,'position6.mat'),'s','d','Trajs','frame')

%%
disp('phase 6')
load(fullfile('.',nameFolder,'position6.mat'))

for dist=120
    for k=1:100
        [s,d,Trajs,frame] = detectTrajArea2(s,d,Trajs,frame,dist,1500,2800);
        [s,d,Trajs,frame] = detectTrajArea3(s,d,Trajs,frame,dist,2600,4000);
        [s,d,Trajs,frame] = detectTrajArea4(s,d,Trajs,frame,dist,3500,5200);
    end
    [s,d,Trajs,frame] = detectTrajsDist(s,d,Trajs,frame,[20 dist],400,1500);
end

save(fullfile('.',nameFolder,'position7.mat'),'s','d','Trajs','frame')

%% Delete trajectories that are too short
disp('phase 7')
load(fullfile('.',nameFolder,'position7.mat'))

[s,frame,Trajs]=deleteShortTraj(s,d,frame,Trajs,25);

save(fullfile('.',nameFolder,'position8.mat'),'s','d','Trajs','frame')

%% Try to link trajectories between different frames
disp('phase 7')
load(fullfile('.',nameFolder,'position8.mat'))

[s,frame,Trajs] = makeTrajectoriesGapTime(s,d,frame,Trajs,120,25);
[s,frame,Trajs] = makeTrajectoriesGapTime(s,d,frame,Trajs,120,2*25);
[s,frame,Trajs] = makeTrajectoriesGapTime(s,d,frame,Trajs,120,4*25);
[s,frame,Trajs] = makeTrajectoriesGapTime(s,d,frame,Trajs,240,4*25);

save(fullfile('.',nameFolder,'position9.mat'),'s','d','Trajs','frame')
%% Check if everything is coherent
% clc;
% checkCoherence(s,d,frame,Trajs)

%% Check if s,s2,frame,frame2,Trajs,Trajs2 are all the same
% compareStruct(d,s,s2,frame,frame2,Trajs,Trajs2)

%% Plot
%tQueue of the trajectory plot
tQueue = 25/25;
%nImages
nImages = d.nImages;

X = zeros(nImages,length(Trajs));
Y = zeros(nImages,length(Trajs));
for jj=1:length(Trajs)
    for ii=1:length(Trajs(jj).Area)
        %For X and Y,the row is the time and the column is the trajectory
        X(Trajs(jj).T(ii),jj) = Trajs(jj).X(ii);
        Y(Trajs(jj).T(ii),jj) = Trajs(jj).Y(ii);
    end
end
% 
% X = zeros(nImages,nBugs);
% Y = zeros(nImages,nBugs);
% for jj=1:length(Trajs)
%     for ii=1:length(Trajs(jj).T)        
%         jZero = find(X(Trajs(jj).T(ii),:)==0,1,'first');
%         X(Trajs(jj).T(ii),jZero) = Trajs(jj).X(ii); 
%         Y(Trajs(jj).T(ii),jZero) = Trajs(jj).Y(ii); 
%     end
% end
for kk=1:nImages
    fullname = fullfile(folder, D(kk).name); %full name of the image
    bw = imread(fullname);
    figure(1);
    imshow(bw)
    hold on
    plot(X(max(1,kk-tQueue*25):kk,:), Y(max(1,kk-tQueue*25):kk,:),'.','MarkerSize',10);%'LineWidth',1);
    title(sprintf('frame=%05.0f', kk))
    %pause(0.5)
    hold off
end