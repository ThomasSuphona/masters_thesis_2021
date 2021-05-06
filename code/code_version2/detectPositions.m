%This function uses the file 'position.mat' in the folder HeuristicPositions, 
%and find the areas that correspond to 2 or more robots, and deduce with
%the orientation and the majoraxislength of the area, where the robots are
%in reality.
%The corrected positions are in the folder HeuristicPositions_2

%The folder in which we read the images
nameFolder = 'New_Conditions_2\1W1000C15B';

userName = 'Everybody';

%load the heuristic positions
name = fullfile('.\HeuristicPositions',nameFolder,'position.mat');
load(name)

%The parent folder
parentFolder = fullfile('C:\Users',userName, ['\Desktop\Quentin_Pikeroen' ...
 '\Quentin_Pikeroen\Image_Sequences\Binarized']);

folder = fullfile(parentFolder,nameFolder);

%calculate the number of images
D=dir([folder '\*.tiff']);
nImages = numel(D); 

%distance
distance = 50;
%nBugs
nBugs = 15;
%tQueue of the trajectory plot
tQueue = 0;
% %matrice index of areas composed of parts of 1 robot
% [image0,index0] = find(Areas < 500 & Areas ~=0);
%matrice index of areas composed of 2 robots exactly
[image,index] = find(Areas > 1600 & Areas <= 2950); %I can't put more than 2800
%matrice index of areas composed of 3 robots exactly
[image3,index3] = find(Areas > 2950); %can't be less than 2950 (or it could detect 3 robots for areas of 2)

for jj=1:nImages
    %index of every small areas
    idx = find(Areas(jj,:) < 600 & Areas(jj,:) ~= 0);
    %index of robots with small area
    for ii=idx
        %if there is an almost big area next to an almost small area
        idxMoy = find(Areas(jj,:) > 600 & Areas(jj,:) < 800 &...
                    (Areas(jj,ii) > 200 & Areas(jj,ii) < 500) );
        idxMoy2 = find( (positionX(jj,idxMoy) - positionX(jj,ii) ).^2 + ...
                     (positionY(jj,idxMoy) - positionY(jj,ii) ).^2   ...
                     < distance^2 );
        if isempty(idxMoy2) == 0
            %disp('a')
            positionX(jj,ii) = 0;
            positionY(jj,ii) = 0;
        end
        %if there is a big area next to the small area, we delete the small area          
        idxBigandSmall = find(Areas(jj,:) >= 600 & Areas(jj,ii) < 200);
        idxBigandSmall2 = find( (positionX(jj,idxBigandSmall) - positionX(jj,ii) ).^2 + ...
                     (positionY(jj,idxBigandSmall) - positionY(jj,ii) ).^2   ...
                     < distance^2 );
        if isempty(idxBigandSmall2)  == 0
            %disp('b')
            positionX(jj,ii) = 0;
            positionY(jj,ii) = 0;
        end
        
        % the index of the closest positions (with a small area)
        idx2 = find( (positionX(jj,idx) - positionX(jj,ii) ).^2 + ...
                     (positionY(jj,idx) - positionY(jj,ii) ).^2   ...
                     < distance^2 ); 
        %change the position of the ii robot to the mean position of every 
        %close area detected
        if isempty(idx(idx2)) == 0 %if the vector is not empty
            %disp('c')
            positionX(jj,ii) = mean(positionX(jj,idx(idx2)));
            positionY(jj,ii) = mean(positionY(jj,idx(idx2)));
        else
            %the positions have already be used as a real robot
            %ATTENTION ici je ne modifie pas les matrices Areas et toutes
            %celles donnees par regionprops : penser à le faire plus tard
            positionX(jj,ii) = 0;
            positionY(jj,ii) = 0;
        end
        %delete the closest positions because they correspond to only 1
        %robot
        idx(idx2) = []; %BE SURE THERE IS NO BUG WITH THE LOOP FOR        
    end
end    
%jj=1;
for jj=1:numel(image)
    %We will imagine a line directed on the orientation of the full region,
    % and set the 2 new positions at a small distance of the whole
    % centroid, onto the line.
    
    %length of the major axis of the area
    hlen = MajorAxisLengths(image(jj),index(jj))/2;
    %orientation of the major axis of the area
    cosOrient = cosd(Orientations(image(jj),index(jj)));
    sinOrient = sind(Orientations(image(jj),index(jj)));
    %new 2 coordinates of the 2 robots
    xcoords = positionX(image(jj),index(jj)) + hlen .* [cosOrient ;-cosOrient]./2;
    ycoords = positionY(image(jj),index(jj)) + hlen .* [-sinOrient ;sinOrient]./2;

%   We change the position matrices with the corrected one (only works with
%   2)
    %we change the position that was in the centroid of the 2 robots
    positionX(image(jj),index(jj)) = xcoords(1);
    positionY(image(jj),index(jj)) = ycoords(1);
    %we change the position that was zero
    positionX( image(jj), find(positionX(image(jj),:)==0,1) ) = xcoords(2);
    positionY( image(jj), find(positionY(image(jj),:)==0,1) ) = ycoords(2);
end

for jj=1:numel(image3)
    %We will imagine a line directed on the orientation of the full region,
    % and set the 3 new positions at a small distance of the whole
    % centroid, onto the line.
    
    %length of the major axis of the area
    hlen = MajorAxisLengths(image3(jj),index3(jj))/2;
    %orientation of the major axis of the area
    cosOrient = cosd(Orientations(image3(jj),index3(jj)));
    sinOrient = sind(Orientations(image3(jj),index3(jj)));
    %new 2 coordinates of the 2 robots
    xcoords = positionX(image3(jj),index3(jj)) + hlen .* [cosOrient ;-cosOrient]./2;
    ycoords = positionY(image3(jj),index3(jj)) + hlen .* [-sinOrient ;sinOrient]./2;
    %We change the position matrices with the corrected one
    %we don't change the position that was in the centroid of the 3 robots
    %we change the positions that was zero
    positionX( image3(jj), find(positionX(image3(jj),:)==0,1)) = xcoords(1);
    positionY( image3(jj), find(positionY(image3(jj),:)==0,1)) = ycoords(1);
    positionX( image3(jj), find(positionX(image3(jj),:)==0,1)) = xcoords(2);
    positionY( image3(jj), find(positionY(image3(jj),:)==0,1)) = ycoords(2);
    
end

%put everything in a  matrix of the size nImages*nBugs (find the 0 inside
%and put them at the end of the line of the matrix)
%kkk=716;
for kk=1:numel(positionX(:,1))
    %find index of zeros
    idxZero = find(positionX(kk,1:nBugs) == 0);
    nZero = numel(idxZero);
    if nZero >= 1
        %find last index of non zeros
        idxNonZero = find(positionX(kk,:)~= 0,nZero,'last');
        %switch the zeros and the non zeros
        positionX(kk,idxZero) = positionX(kk,idxNonZero);
        positionX(kk,idxNonZero) = 0;
        positionY(kk,idxZero) = positionY(kk,idxNonZero);
        positionY(kk,idxNonZero) = 0;
    end
end
%disp(positionX(kk,:))

%if the last columns are full of zeros, we delete them
if isempty(find(positionX(:,nBugs+1:end),1))
    positionX(:,nBugs+1:end) = [];
    positionY(:,nBugs+1:end) = [];    
end

%% Save the positions of the particles

saveFolder = fullfile('.\HeuristicPositions_2',nameFolder);

%create a folder in which the heuristic positions will be saved
mkdir(saveFolder);

%save the heuristic positions in the right folder
save('position.mat', 'positionX', 'positionY','Areas','Orientations',...
    'MinorAxisLengths','MajorAxisLengths')  
movefile('position.mat', saveFolder);

%% Show the result 
% fullname = fullfile(folder, D(image(jj)).name); %full name of the image
% bw = imread(fullname);
% imshow(bw)
% hold on
% plot(xcoords, ycoords, 'r.', 'MarkerSize',20);
% hold off

%kkk=711;
for kkk=3212:3212%nImages%transpose(find(positionX(:,17) ~= 0))
    fullname = fullfile(folder, D(kkk).name); %full name of the image
    bw = imread(fullname);
    imshow(bw)
    hold on
    
%     hlen = MajorAxisLengths(kkk,:)/2;
%     cosOrient = cosd(Orientations(kkk,:));
%     sinOrient = sind(Orientations(kkk,:));
%     xcoords = positionX(kkk,:)+ hlen .* [cosOrient ;-cosOrient];
%     ycoords = positionY(kkk,:) + hlen .* [-sinOrient ;sinOrient];
%     pl = line(xcoords, ycoords,'Color','blue','LineStyle','-','LineWidth',4);
    
    plot(positionX(max(1,kkk-25*2*tQueue:kkk),:), positionY(max(1,kkk-25*2*tQueue:kkk),:), 'r.', 'MarkerSize',10); 
    
    %F(kkk) = getframe;
    drawnow
    %pause(0.5)
    hold off

end
