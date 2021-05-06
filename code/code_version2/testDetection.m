%This function detects every regions in only one image, and plot the
%detected region onto the imag

load('undistort.mat')
%% parameters

%to detect cylinders %mean(areas) = 38
minArea = 15;
maxArea = 65;
%threshold = 230;
threshold = 205;

%to detect bugs %mean(area) = 760
% minArea = 400;
% maxArea = 12000;
% threshold = 245;

%img = imread('C:\Users\Everybody\Downoads\My photo - 6-19-2019_4-16-22---.jpg');
%img = imread('C:\Users\Everybody\Desktop\Quentin_Pikeroen\Quentin_Pikeroen\Videos\TestDetectionImage_2.jpg');
img = imread(['C:\Users\parti\Downloads\'...
'My photo - 2019-07-18-16-26-55.jpg']);

%% crop and binarize
%transform
img = undistortImage(img,cameraParams_2);

img = imcrop(img,[320,0,1270,1510]);
%img = imcrop(img,[425,58,1085,957]);
%binarize the image
I = rgb2gray(img); %make the image in gray scale
I = imcomplement(I); %invert black and white
I = I > threshold; %binarize





%% Detect centroids
stats = regionprops(I,'Area','Centroid');

%put the 2 stats parameters in 3 matrices 
areas = cat(1, stats.Area); %matrix N*1
centroids = cat(1,stats.Centroid); %matrix 2N*1

%index of elements that have a bigger area than the threshold
index = ((areas > minArea) & (areas < maxArea)); 

%Delete the areas and centroids we don't want
areas = areas(index);
centroids = centroids(index,:);

disp(numel(areas))
%% Detect circles
% [centers,radii] = imfindcircles(img,[20 100],'ObjectPolarity','dark', ...
%     'Sensitivity',0.99)
% imshow(img)
% h = viscircles(centers,radii);


%% Plot 
%show the binarized image
figure(1)
imshow(I);


%show the input image with the centroids
figure(2)
imshow(img)
hold on
    img = plot(centroids(:,1), centroids(:,2), 'r.', 'MarkerSize',10);  
hold off