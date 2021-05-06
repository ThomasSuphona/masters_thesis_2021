close all;clear all;

%% Parameters
%folder name
names = {'0W1000C15B','1W1000C15B'};
%names={'0W0C16B','0W300C15B','0W700C15B','1W300C15B','1W700C15B','2W700C15B'};
%nameFolder = 'New_Conditions_2\0W0C16B';
%use name
userName = 'Everybody';

%kk=1;
for kk = 1:length(names)
tic
% load the positions given by the detectTrajectories function
fullname = strcat('.\Positions\New_Conditions_2\',names{kk},'\','position.mat');
load(fullname)


%%
%The parent folder
parentFolder = fullfile('C:\Users',userName, ['\Desktop\Quentin_Pikeroen' ...
 '\Quentin_Pikeroen\Image_Sequences\Binarized']);

folder = fullfile(parentFolder,'New_Conditions_2',names{kk});

D=dir([folder '\*.tiff']);
%Here bw is used to have access to the dimensions of the image
bw = imread(fullfile(folder, D(1).name)); 

%nImages
nImages = numel(D);
%nBugs
nBugs = numel(X(1,:));
%nPoints
nPoints = 1e2;
%Radius of the points
R = 7;
%decreasing parameter
alpha = exp(-(1:nPoints)/30);%fliplr(0.01:0.01:1);

lx = length(bw(1,:));
ly = length(bw(:,1));
image = zeros(ly,lx);
[mx,my] = meshgrid(1:lx,1:ly);
%F = struct('cdata', uint8((zeros(lx,ly,3)))*nImages,'colormap',[]*nImages);
% create the video writer
writerObj = VideoWriter(strcat('..\Trajectories','\',names{kk},'.avi'));
% set the seconds per image
writerObj.FrameRate = 30;

% open the video writer
open(writerObj);
%nImages= 100;
for ii=1:nImages
    %FIND THE BEST WAY FOR IMAGE TO DECREASE
    image = max(image-(1e-3),0);%1 - exp(-image*1.2);
    for jj=1:nBugs
        image(((mx-X(ii,jj)).^2+(my-Y(ii,jj)).^2) < R^2) = min(1.2, ...
  0.07 + image(((mx-X(ii,jj)).^2+(my-Y(ii,jj)).^2) < R^2)      );
    end 

%read the binarized image
fullname = fullfile(folder, D(ii).name); %full name of the image
bw = imread(fullname);  %zeros(ly,lx); 
bw = 1 - bw;
%transform the black and white image into a color image
bw = repmat(bw,[1,1,3]).*1;
%add or delete 'image' to the matrix bw, in order to increase the blue
bw(:,:,3) = bw(:,:,3) + image;
bw(:,:,2) = bw(:,:,2) - image;
bw(:,:,1) = bw(:,:,1) - image;

bw(bw >1) = 1;
bw(bw<0) = 0;
% figure(1);
% img = imshow(bw);
%write the image to the video
writeVideo(writerObj, bw);
%F(ii) = getframe(gca);
% hold on
% %imshow(rgbImage)
% hold off

end

%%
% % create the video writer
% writerObj = VideoWriter(strcat('..\Trajectories','\',names{kk},'.avi'));
% % set the seconds per image
% writerObj.FrameRate = 30;
% 
% % open the video writer
% open(writerObj);

% for ii=1:length(F)
%     % convert the image to a frame
%     frame = F(ii) ;    
%     writeVideo(writerObj, frame);
% end
% close the writer object
close(writerObj);
toc
end