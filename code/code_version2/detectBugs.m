% Detect the position of each robots in each frames, but doesn't extract
% their trajectories
%
%To execute the function you need to change the nameFolder, and the
%estimated number of bugs (nBugs).
%
%
tic
%% Parameters

%The folders in which are the images
% names = {'input_videos\C5B838P2M','input_videos\C5B838P3M',...
%     'input_videos\C10B838P0M','input_videos\C10B838P1M','input_videos\C10B838P2M'...
%     'input_videos\C10B838P3M','input_videos\C15B838P0M','input_videos\C15B838P1M',...
%     'input_videos\C15B838P2M','input_videos\C15B838P3M',...
%     'input_videos\C20B838P0M','input_videos\C20B838P1M','input_videos\C20B838P2M',...
%     'input_videos\C20B838P3M','input_videos\C25B838P0M','input_videos\C25B838P1M',...
%     'input_videos\C25B838P2M','input_videos\C25B838P3M','input_videos\C30B838P0M',...
%     'input_videos\C30B838P1M','input_videos\C30B838P2M','input_videos\C30B838P3M'};

names = {'New_Conditions_2\1W1000C15B'};

userName = 'Everybody';

% %The number of bugs (is changed after each detection)
% nBugs = 15;

%Min and Max Area we keep for the detection
minArea = 100;
maxArea = 1e4;
%maxArea = 1300;

%%

for ii=1:numel(names)
   
%The number of bugs determined by the name of the folder    
for jj=1:30
    if contains(names{ii}, strcat(num2str(jj), 'B'))
        nBugs = jj; 
    end
end
    
%The parent folder
parentFolder = fullfile('C:\Users',userName, ['\Desktop\Quentin_Pikeroen' ...
 '\Quentin_Pikeroen\Image_Sequences\Binarized']);

%The folder in which we read the images
nameFolder = names{ii};

folder = fullfile(parentFolder,nameFolder);

%calculate the number of images
D=dir([folder '\*.tiff']);
nImages = numel(D); 

%successive positions of the robots
positionX = zeros(nImages, nBugs);
positionY = zeros(nImages, nBugs);

%successive areas of the robots
Areas = zeros(nImages, nBugs); 
Orientations = zeros(nImages, nBugs);
MinorAxisLengths = zeros(nImages, nBugs); 
MajorAxisLengths = zeros(nImages, nBugs); 

%% Detect regions

disp(nImages)
%kk = 1888;
for kk = 1:nImages
    fullname = fullfile(folder, D(kk).name); %full name of the image
    bw = imread(fullname);

    stats = regionprops(bw,'Area','Centroid','Orientation',...
        'MajorAxisLength','MinorAxisLength');

    %put the 2 stats parameters in 3 matrices 
    areas = cat(1, stats.Area); %matrix N*1
    centroids = cat(1,stats.Centroid); %matrix 2N*1
    orientations = cat(1,stats.Orientation); %matrix N*1
    majorAxisLengths = cat(1, stats.MajorAxisLength); %matrix N*1
    minorAxisLengths = cat(1, stats.MinorAxisLength); %matrix N*1

    %index of elements that have a bigger area than the threshold
    index = ((areas > minArea) & (areas < maxArea)); 

    %Delete the areas and centroids we don't want
    areas = areas(index);
    centroids = centroids(index,:);
    orientations = orientations(index);
    majorAxisLengths = majorAxisLengths(index);
    minorAxisLengths = minorAxisLengths(index);
   
%     %Show the centroids and directions of every detected robots
%     imshow(bw)
%     hold on
%     
%     hlen = transpose(majorAxisLengths(:))/2;
%     cosOrient = cosd(transpose(orientations(:)));
%     sinOrient = sind(transpose(orientations(:)));
%     xcoords = transpose(centroids(:,1))+ hlen .* [cosOrient ;-cosOrient];
%     ycoords = transpose(centroids(:,2)) + hlen .* [-sinOrient ;sinOrient];
%     pl = line(xcoords, ycoords,'Color','blue','LineStyle','-','LineWidth',3);
%     %pl.Color(:) = 'blue';
%     %pl.LineStyle = '-';
%     %pl.LineWidth = 3;
%     
%     img = plot(centroids(:,1), centroids(:,2), 'r.', 'MarkerSize',20);
%     drawnow
%     hold off

    nBugs = numel(areas); %Be careful nBugs changes here

    % Register the coordinates in the matrix position
    newCentroids = permute(centroids, [3 1 2]);

    positionX(kk, 1:nBugs) = newCentroids(1, :, 1);
    positionY(kk, 1:nBugs) = newCentroids(1, :, 2);
    
    Areas(kk,1:nBugs) = transpose(areas);
    Orientations(kk,1:nBugs) = transpose(orientations);
    MajorAxisLengths(kk,1:nBugs) = transpose(majorAxisLengths);
    MinorAxisLengths(kk,1:nBugs) = transpose(minorAxisLengths);
    
    if floor(kk/200)==kk/200 
        disp(kk) 
    end
end %of the Images loop


%% Save the positions of the particles

saveFolder = fullfile('.\HeuristicPositions',nameFolder);

%create a folder in which the heuristic positions will be saved
mkdir(saveFolder);

%save the heuristic positions in the right folder
save('position.mat', 'positionX', 'positionY','Areas','Orientations',...
    'MinorAxisLengths','MajorAxisLengths') 
%save('positionY.mat') 
movefile('position.mat', saveFolder);
%movefile('positionY.mat', saveFolder);

end %of the Names loop

%% Plot the trajectories (which are wrong with several robots)

% t=(1:nImages)/25;
% 
% figure(1)
% plot(t,positionX(1:nImages,1),t,positionY(1:nImages,1));
% 
% figure(2)
% %imshow(bw)
% hold on
% plot(positionX(1:nImages,1),positionY(1:nImages,1));
% hold off

toc 
%% Plot
for kkk=1:nImages       
    fullname = fullfile(folder, D(kkk).name); %full name of the image
    bw = imread(fullname);
    imshow(bw)
    hold on
    
    hlen = MajorAxisLengths(kkk,:)/2;
    cosOrient = cosd(Orientations(kkk,:));
    sinOrient = sind(Orientations(kkk,:));
    xcoords = positionX(kkk,:)+ hlen .* [cosOrient ;-cosOrient];
    ycoords = positionY(kkk,:) + hlen .* [-sinOrient ;sinOrient];
    pl = line(xcoords, ycoords,'Color','blue','LineStyle','-','LineWidth',4);
    
    plot(positionX(max(1,kkk-25*2:kkk),:), positionY(max(1,kkk-25*2:kkk),:), 'r.', 'MarkerSize',10); 
    
    %F(kkk) = getframe;
    drawnow

    hold off

end

%% Create a video
% % create the video writer with 1 fps
% writerObj = VideoWriter('myVideo_2Bugs.avi');
% % set the seconds per image
% writerObj.FrameRate = 25;
% 
% % open the video writer
% open(writerObj);
% % write the frames to the video
% for iii=1:length(F)
%     % convert the image to a frame
%     frame = F(iii) ;    
%     writeVideo(writerObj, frame);
% end
% % close the writer object
% close(writerObj);

%% tests
% %Test
%     mkdir('Test');
%     saveas( img, [pwd strcat( '\Test\', sprintf('%g', kk) ) ] )

%Disp rectangles around the detected objetcs
% for kk = 1:numObj
%     rectangle('Position', [boundingBox(kk,1),boundingBox(kk,2),boundingBox(kk,3),...
%     boundingBox(kk,4)],'EdgeColor','r','LineWidth',2 )
% end

% Detect circles
% [centers,radii] = imfindcircles(img,[20 100],'ObjectPolarity','dark', ...
%     'Sensitivity',0.99)
% imshow(img)
% h = viscircles(centers,radii); 