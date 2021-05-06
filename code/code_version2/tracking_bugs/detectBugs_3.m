% Detect the positions of each robots in each frames, but doesn't extract
% their trajectories
%
%It creates a struct array 's' with fields :
% s.Area
% s.X
% s.Y
% s.Orientation
% s.MajorAxisLength
% s.MinorAxisLength
% s.Eccentricity
% s.isTraj
% where (for example) s(i).X is an array with all the X positions of 
% all the robots in the frame number i.
%
%It also creates a struct array 'd' with fields :
% d.nImages      the number of images
% d.nBugs        the number of robots
% d.nCylinders   the number of obstacles
% d.nWeigths     the number of weights
%
%To execute the function you need to change the name of the folder in which
%the images are read

%tic
clear variables, close all
% Parameters

%The parent folder
parentFolder = fullfile('..\..\Image_Sequences\Binarized\1RobotControlExp');

%The folder which contains the folders with the image sequences
listing = dir(fullfile(parentFolder));
%names = {'0W300C15B'};

%Min and Max Area we keep for the detection
minArea = 100;
maxArea = 1e4;
%maxArea = 1300;

%Loop through all the folders that contains folders with binarized images

for ii=3:numel(listing)
    
%The name of the folder in which we read the images
name = listing(ii).name;

%The number of bugs determined by the name of the folder    
for jj=[1,2,5:5:25]
    if contains(name, strcat(num2str(jj), 'B'))
        nBugs = jj; 
    end
end
for jj=0:100:1300
    if contains(name, strcat(num2str(jj), 'C'))
        nCylinders = jj; 
    end
end
for jj=0:3
    if contains(name, strcat(num2str(jj), 'W'))
        nWeigths = jj; 
    end
end
    


% %The folder in which we read the images
% nameFolder = fullfile('New_Conditions_2',name);

folder = fullfile(parentFolder,name);

%calculate the number of images
D=dir([folder '\*.tiff']);
nImages = numel(D); 

%structure array
clear s d
s(nImages) = struct('Area',[],'X',[],'Y',[],'Orientation',[],...
    'MajorAxisLength',[],'MinorAxisLength',[],'Eccentricity',[],'isTraj',[]);
d.nImages = nImages;
d.nBugs = nBugs;
d.nCylinders = nCylinders;
d.nWeigths = nWeigths;

%% Detect regions

disp(nImages)

%Loop through all the images in one folder

%kk = 1;
for kk = 1:nImages
    %We read the binarized image
    fullname = fullfile(folder, D(kk).name); %full name of the image
    bw = imread(fullname);
    
    %Detect all the regions and put the data into a struct array
    stats = regionprops(bw,'Area','Centroid','Orientation',...
        'MajorAxisLength','MinorAxisLength','Eccentricity');
    
    %names of the field of stats
    fn = fieldnames(stats);
    %the struct array 's' takes all the values of 'stats', exept for 
    %stats.Centroid and for s.isTraj
    for iii=[1,3:numel(fn)]
        s(kk).(fn{iii}) = [stats.(fn{iii})];
    end
    %s.X and s.Y take the values of stats.Centroid
    M = [stats.Centroid];
    s(kk).X = M(1:2:end);%odd values
    s(kk).Y = M(2:2:end);%even values
    
    %the field isTraj takes the value 0 for every positions
    s(kk).isTraj(1:length(s(kk).Area)) = 0;

    %index of elements that have a bigger area than the threshold
    index = ( (s(kk).Area > minArea) & (s(kk).Area < maxArea) ); 
    
    %For each field array,delete the robots that don't have the 
    %wanted area
    fd = fieldnames(s);
    for iii=1:numel(fd)  
        matrix = s(kk).(fd{iii});
        [s(kk).(fd{iii})] = matrix(index); 
    end
    
    %Disp the number of frame processed every 200 frames
    if floor(kk/200)==kk/200 
        disp(kk) 
    end
end %of the Images loop


%% Save the positions of the particles

saveFolder = fullfile('.',name);

%create a folder in which the heuristic positions will be saved
mkdir(saveFolder);

%save the heuristic positions in the right folder
save('position.mat', 's','d') 
%save('positionY.mat') 
movefile('position.mat', saveFolder);
%movefile('positionY.mat', saveFolder);

end %of the Names loop
%toc 
%% Plot
% for kk=1:2       
%     fullname = fullfile(folder, D(kk).name); %full name of the image
%     bw = imread(fullname);
%     imshow(bw)
%     hold on
%     
%     hlen = s(kk).MajorAxisLength/2;
%     cosOrient = cosd(s(kk).Orientation);
%     sinOrient = sind(s(kk).Orientation);
%     xcoords = s(kk).X + hlen .* [cosOrient ;-cosOrient];
%     ycoords = s(kk).Y + hlen .* [-sinOrient ;sinOrient];
%     pl = line(xcoords, ycoords,'Color','blue','LineStyle','-','LineWidth',4);
%     
%     plot(s(kk).X, s(kk).Y,'r.', 'MarkerSize',10); 
%     axis = 'on';
%     %F(kkk) = getframe;
%     drawnow
% 
%     hold off
% 
% end
