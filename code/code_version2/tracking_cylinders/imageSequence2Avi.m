%transform the undistorted image sequences into an avi video

%name of the image sequence folder
names={'0W700C25B'};

%crop of the images
rect = [330 0 1265 1075]; %new_conditions_2
   
ii=1;

%Name of the folder 
folderName = strcat('New_Conditions_2\', names{ii});

userName = 'Everybody'; 

%the directory of the image sequence
imageSequencesDir = fullfile('C:\Users',userName,['\Desktop\Quentin_Pikeroen' ...
'\Quentin_Pikeroen\Image_Sequences\Undistorted']);

%the directory where the video will be written
videosDir = fullfile('C:\Users',userName,['\Desktop\Quentin_Pikeroen' ...
'\Quentin_Pikeroen\UndistortedVideos']);

%Cell of every image names
imageNames = dir(fullfile(imageSequencesDir,folderName,'*.jpg'));
imageNames = {imageNames.name}';


outputVideo = VideoWriter(strcat(videosDir,'\',names{ii},'.avi'));
outputVideo.FrameRate = 30;
open(outputVideo)

%Read, crop and write all images
for ii = 1:length(imageNames) 
   %read the image
   img = imread(fullfile(imageSequencesDir,folderName,imageNames{ii}));
   %crop
   img = imcrop(img, rect);
   %write
   writeVideo(outputVideo,img)
end

close(outputVideo)