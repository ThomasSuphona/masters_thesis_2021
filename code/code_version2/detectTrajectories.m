% This program detects the trajectories of several particles
% It uses the matrices positionX and positionY given by the function
% dectectPositions, and extracts the trajectories of each robots.
%It saves the X and Y position in the folder Positions

close all

%% Parameters

%folder name
nameFolder = 'New_Conditions_2\1W1000C15B';

% load the positions given by the detectBugs function
fullname = strcat('.\HeuristicPositions_2\',nameFolder,'\','position.mat');
load(fullname)

userName = 'Everybody';
%number of images (if nothing it will process every images of the folder
% N = 1367;
N = numel(positionX(:,1)); 

%number of robots
n = 15; 

%The maximum distance for which he will try to find the same robot in the
%next frame (in pixels)
d = 180;

%%

%The parent folder
parentFolder = fullfile('C:\Users',userName, ['\Desktop\Quentin_Pikeroen' ...
 '\Quentin_Pikeroen\Image_Sequences\Binarized']);

folder = fullfile(parentFolder,nameFolder);

D=dir([folder '\*.tiff']);

%The image shown at the end of the figure 2
img = fullfile(['C:\Users\',userName,'\Desktop\Quentin_Pikeroen\Quentin_Pikeroen\'...
'Image_Sequences\Binarized'],nameFolder,D(N).name);
bw = imread(img);

%The new positions
newX = zeros(N,n);
newY = zeros(N,n);

testX = positionX(1:N,1:n);
testY = positionY(1:N,1:n);

newX(1,:) = testX(1,:);
newY(1,:) = testY(1,:);

%Le but ? partir de ces tests est de trouver les trajectoires de chacune
%des particules.

for ii=1:N-1 %pour toutes les images
    %ii
    %si une position n'est pas detectee (donc vaut 0 dans la matrice testX) 
    if ismember(0,testX(ii+1,1:n))
           index = find(testX(ii+1,1:n)==0);
           %we put an infinite value so this coordonate will never be
           %inside the distance and will never be considered.
           testX(ii+1,index) = inf;
    end

    for jj=1:n %pour tous les robots
        %jj
 
        %distances entre tous les robots et le jj-ieme 
        %distance est modifie a chaque boucle
        distance = sqrt(  ( testX(ii+1,1:n)-newX(ii,jj) ).^2 +...
                          ( testY(ii+1,1:n)-newY(ii,jj) ).^2     );
                 
        %s'il ne trouve aucune distance proche (particule disparue ou        
        %bien mixage de 2 particules
        if (distance < d) == 0 
            %vitesse decroissante (vitesse divisee par 3 a chaque etape)            
            newX(ii+1,jj) = newX(ii,jj) +...
                            ( newX(ii,jj)-newX(max(1,ii-1),jj)+...
                             newX(max(1,ii-1),jj)-newX(max(1,ii-2),jj)+...
                             newX(max(1,ii-2),jj)-newX(max(1,ii-3),jj)+...
                             newX(max(1,ii-3),jj)-newX(max(1,ii-4),jj) )/(3*4);
            newY(ii+1,jj) = newY(ii,jj) +...        
                            ( newY(ii,jj)-newY(max(1,ii-1),jj)+...
                              newY(max(1,ii-1),jj)-newY(max(1,ii-2),jj)+...
                              newY(max(1,ii-2),jj)-newY(max(1,ii-3),jj)+...
                              newY(max(1,ii-3),jj)-newY(max(1,ii-4),jj) )/(3*4);
            %garde la meme position
            %newX(ii+1,jj) = newX(ii,jj);
            %newY(ii+1,jj) = newY(ii,jj);    
            %disp('positionConservee')
        
        %s'il trouve plusieurs distances au lieu d'une seule
        elseif numel(find(distance < d)) >= 2
            [M,I] = min(distance);
            %il choisit la distance la plus proche
            newX(ii+1,jj) = testX(ii+1, I);
            newY(ii+1,jj) = testY(ii+1, I);
            %le robot jj ne sera pas choisi par un autre pour cette image
            testY(ii+1,I) = Inf; 
            
            %disp('2 distances trouvees')
        
        %Il n'existe qu'une seule distance ou il trouve la particule    
        else     
            %nouvelle position du jj-ieme robot
            newX(ii+1,jj) = testX(ii+1,(distance < d));
            newY(ii+1,jj) = testY(ii+1,(distance < d));
            %le robot jj ne sera pas choisi par un autre pour cette image
            testY(ii+1,(distance < d)) = Inf;
        end                
    end
end
%% Plot the trajectories
t=(1:N)/30;
%plot X and Y of every robots with time
trajs = 1:n;
figure(1)
hold on
%for j=1:n
plot(t,newX(1:N,trajs),t,newY(1:N,trajs));
%end
hold off

%plot X in funciton of Y 
trajs = 1:n;%15;
figure(2)
imshow(bw)
hold on
plot(newX(1:N,trajs),newY(1:N,trajs), 'Marker', '.', 'MarkerSize', 5,'Linewidth', 1);
plot(newX(N,trajs),newY(N,trajs), 'Marker','.','MarkerSize',30,'Color','r')
hold off

%% Save the trajectories
X = newX;
Y = newY;

saveFolder = fullfile('.\Positions',nameFolder);

%create a folder in which the heuristic positions will be saved
mkdir(saveFolder);

% %save the heuristic positions in the right folder 
% save('X.mat');
% save('Y.mat');
% movefile('X.mat', saveFolder);
% movefile('Y.mat', saveFolder);


%save the positions in the right folder
save('position.mat', 'X', 'Y','Areas','Orientations',...
    'MinorAxisLengths','MajorAxisLengths')  
movefile('position.mat', saveFolder);


%% show the trajectories in the binarized video 

F = struct('cdata', uint8((zeros(717,844,3)))*nImages,'colormap',[]*nImages);
for kkk=1:nImages       
    fullname = fullfile(folder, D(kkk).name); %full name of the image
    bw = imread(fullname);
    figure(3), imshow(bw)
    hold on
    
%     hlen = MajorAxisLengths(kkk,:)/2;
%     cosOrient = cosd(Orientations(kkk,:));
%     sinOrient = sind(Orientations(kkk,:));
%     xcoords = newX(kkk,:)+ hlen .* [cosOrient ;-cosOrient];
%     ycoords = newY(kkk,:) + hlen .* [-sinOrient ;sinOrient];
%     pl = line(xcoords, ycoords,'Color','blue','LineStyle','-','LineWidth',4);

    %plot(newX(max(1,kkk-25*2:kkk),:), newY(max(1,kkk-25*2:kkk),:), 'r.', 'MarkerSize',10); 
    plot(newX(max(1,kkk-25*4:kkk),1:n),newY(max(1,kkk-25*4:kkk),1:n), 'Marker', '.', 'MarkerSize', 5,'Linewidth', 1);
    %F(kkk) = getframe;
    %drawnow
    %pause(1)
    hold off

end

% %% Create a video
% % create the video writer with 1 fps
% writerObj = VideoWriter('myVideo_10Bugs.avi');
% % set the seconds per image
% writerObj.FrameRate = 25;
% 
% % open the video writer
% open(writerObj);
% % write the frames to the video
% for kkk=1:length(F)
%     kkk
%     % convert the image to a frame
%     frame = F(kkk) ;    
%     writeVideo(writerObj, frame);
% end
% % close the writer object
% close(writerObj);
