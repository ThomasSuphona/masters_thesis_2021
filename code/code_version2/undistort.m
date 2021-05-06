%This function transform a video into a sequence of images, and undistort
%every images.
%
%The input videos are in the folder \Videos, and the output videos are in
%the folder \Image_Sequences.
%To execute the program you need to change the folderName, the extension
%of the video (avi, mov, mp4, webm,...), and the user name (Saga or
%Everybody)
clc
clear all

load('undistort.mat')

%%Folder(s) to analyse
% names = {'Artemis','Baldr', 'Cassandre', 'Clementine', 'E-Coli', 'Elise', 'Fatima'...
%     'Freya', 'Freyja', 'Geoffroy', 'Giselle', 'Hecate', 'Hera', 'Iseult'...
%     'Jean', 'Loki', 'Nemesis', 'Odin', 'Pierre', 'Poseidon', 'Salomon'...
%     'Sarah', 'Sauron', 'Simon', 'Solene', 'Stephane', 'Sylvain', 'Sylvia'...
%     'Tristan', 'Zeus'};
% names = {'5_Bugs','10_Bugs','15_Bugs', '20_Bugs', '25_Bugs','30_Bugs'};
% names = {'C15B838P0M',...
%     'C15B838P1M','C15B838P2M','C15B838P3M','C20B838P0M','C20B838P1M',...
%     'C20B838P2M','C20B838P3M','C25B838P0M','C25B838P1M','C25B838P2M'...
%     'C25B838P3M','C30B838P0M','C30B838P1M','C30B838P2M','C30B838P3M'};
%     
% names = {'5B100P1M','5B200P1M','5B300P1M','5B400P1M','5B500P1M',...
%     '5B600P1M','5B700P1M','5B800P1M','10B100P1M','10B200P1M',...
%     '10B300P1M','10B400P1M','10B500P1M','10B600P1M','10B700P1M','10B800P1M',...
%     '15B100P1M','15B200P1M','15B300P1M','15B400P1M','15B500P1M','15B600P1M',...
%     '15B700P1M','15B800P1M',};

% names = {'0W600C1B','0W600C2B','0W600C5B','0W600C10B',...
%     '0W600C15B','0W600C20B','0W600C25B','0W700C1B','0W700C2B','0W700C5B',...
%     '0W700C10B','0W700C15B','0W700C20B','0W700C25B','1W100C1B','1W100C2B',...
%     '1W100C5B','1W100C10B','1W100C15B','1W100C20B','1W100C25B','1W200C1B',...
%     '1W200C2B','1W200C5B','1W200C10B','1W200C15B','1W200C20B','1W200C25B',...
%     '1W300C1B','1W300C2B','1W300C5B','1W300C10B','1W300C15B','1W300C20B','1W300C25B',...
%     '1W400C1B','1W400C2B','1W400C5B','1W400C10B','1W400C15B','1W400C20B','1W400C25B',...
%     '1W500C1B','1W500C2B','1W500C5B','1W500C10B','1W500C15B','1W500C20B','1W500C25B',...
%     '1W600C1B','1W600C2B','1W600C5B','1W600C10B','1W600C15B','1W600C20B','1W600C25B',...
%     '1W700C1B','1W700C2B','1W700C5B','1W700C10B','1W700C15B','1W700C20B','1W700C25B',...
%     '2W100C1B','2W100C2B','2W100C5B','2W100C10B','2W100C15B','2W100C20B','2W100C25B',...
%     '2W200C1B','2W200C2B','2W200C5B','2W200C10B','2W200C15B','2W200C20B','2W200C25B',};

names = {'1W1300C1B'};

%names = {'0W300C15B'};

for jj=1:numel(names)

%Name of the folder 
folderName = names{jj};
%folderName = 'New_Conditions_2\0P\1B0P0M';

extension = '.mkv';

%% Video to analyse 

% %load the transormation for the images
% load('tform.mat')

videosDir =  fullfile('..\Videos_Cut');

inputDir = strcat(videosDir, '\', folderName, extension)
video = VideoReader(inputDir); %read the video

%% Create the new directories for the image sequence

imageSequencesDir = fullfile('..\Image_Sequences\Undistorted');

%makes a new directory under the parent
mkdir(imageSequencesDir, folderName) 
%str name of the output directory
outputDir = fullfile(imageSequencesDir, folderName);

%% Count the number of images in the video
Nframes = sprintf('%04.0f', video.Duration*video.FrameRate)

%% Create the image sequence

ii = 1;
while hasFrame(video) %determine if the frame is available to read, 
                       %and return a logical 1 or 0
    img = readFrame(video); %read the next available videoframe

    %undistort the image
%     %crop
%     rect = [400  30 1500-400 1000-30]; %[xmin ymin width height]
%     img = imcrop(img, rect);
    
    %transform 
    %img = imwarp(img, tform); %The older tranformation
    img = undistortImage(img,cameraParams_2);

    %write the image in the output folder
    filename = [sprintf('%05.0f', ii), '.jpg']; %name of the frame
    fullname = fullfile(outputDir, filename); %return full file name
    imwrite(img, fullname) %write out to a JPEG file

    ii=ii+1;
    if floor(ii/200)==ii/200 
        disp(ii) 
    end
end

end