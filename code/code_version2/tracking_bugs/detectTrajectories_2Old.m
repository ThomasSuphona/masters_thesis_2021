clear all, close all
%The folder in which we read the images
nameFolder = 'New_Conditions_2\1W800C15B';

%load the heuristic positions
name = fullfile('.',nameFolder,'position.mat');
load(name)

%Folders of the binarized image sequences
parentFolder = fullfile('..\..\Image_Sequences\Binarized');
folder = fullfile(parentFolder,nameFolder);
%Images folder
D=dir([folder '\*.tiff']);

%number of images
nImages = d.nImages;
%number of bugs
nBugs = d.nBugs;



%distance
dist = 12;
% %matrice index of areas composed of parts of 1 robot
% [image0,index0] = find(Areas < 500 & Areas ~=0);
% %matrice index of areas composed of 2 robots exactly
% [image,index] = find(Areas > 1600 & Areas <= 2950); %I can't put more than 2800
% %matrice index of areas composed of 3 robots exactly
% [image3,index3] = find(Areas > 2950); %can't be less than 2950 (or it could detect 3 robots for areas of 2)

%% Initialisation
%s.isTraj = -1 if belongs to a trajectory
%            0 if doesn't belong to a trajectory
%            j if it is the first point of the the j-th trajectory
%frame(ii).indexTraj is a vector with the index of the trajectories
%that end at the frame ii
%struct array Trajs
Trajs = struct('Area',[],'X',[],'Y',[],'Orientation',[],...
    'MajorAxisLength',[],'MinorAxisLength',[],'Eccentricity',[],...
        'T',[],'P',[]);
fn = fieldnames(Trajs);
%indexes of points for which we will try to construct a trajectory
for ii=1:nImages
    frame(ii).indexTraj = [];
end
%Create the first point of the trajectories
for jj=1:length(s(1).Area) 
    if s(1).Area(jj) > 700 && s(1).Area(jj) < 1300 
        n = length(Trajs);
        for jjj=1:numel(fn)-2
            %For each trajectory in Trajs, the first point of the
            %trajectory takes the value of one point (jj) of the first 
            %frame of the struct 's'
            Trajs(n+1).(fn{jjj})(1) = s(1).(fn{jjj})(jj);
        end
        %Time of the first point of trajectories
        Trajs(n+1).T(1) = 1;
        %position of the robot in the struct array s(ii)
        Trajs(n+1).P(1) = jj;
        %the first points belongs now to a trajectory
        s(1).isTraj(jj) = jj;
        %We add the index of the robot in s(1) into indexTraj
        frame(1).indexTraj = [frame(1).indexTraj,jj];
    end    
end
%remove the first element of Trajs, wich contains nothing
Trajs(1) = [];

%% Loop 1
for dist = [12 16 20]
for ii=1:nImages-1 %for each frame
    %trajs to delete in the frame.indexTraj
    deleteTraj = [];
    %trajs to delete in the struct Trajs
    deleteIdxTraj = [];
    %For all trajectories in construction
    for jj=frame(ii).indexTraj
        %The index in the struct array s(ii)
        idx=Trajs(jj).P(end);
        %Find the closest point inside the circle of diameter d=dist,
        %which doesn't belong to a trajectory,
        %and which has an area between 700 and 1300
        R = (s(ii).X(idx)-s(ii+1).X).^2 + (s(ii).Y(idx)-s(ii+1).Y).^2;
        logical1 = sqrt(R) < dist; 
        %points that belong to no trajectory or to the first point of a traj
        logical2 = s(ii+1).isTraj >= 0; 
        logical3 = s(ii+1).Area > 700 & s(ii+1).Area < 1300;
            
        %Index of the closest point that correspond to criteria
        index = find(logical1 & logical2 & logical3);
        [~,index2] = min(R(index));
        
        %We add the new point to the trajectory
        if isempty(index2) == 0
            deleteTraj = [deleteTraj,find(frame(ii).indexTraj==jj)];
            %if the choosen point belongs to no trajectory
            if s(ii+1).isTraj(index(index2)) == 0
                for jjj=1:numel(fn)-2
                    %add to the trajectory jj the point in s(ii+1) that we
                    %choosed, for each common field jjj of 'Trajs' and 's' 
                    %[~,index2] = min(R(index))
                    Trajs(jj).(fn{jjj})(end+1) = s(ii+1).(fn{jjj})(index(index2));
                end
                %the time of the point
                Trajs(jj).T(end+1) = ii+1;
                %the position of the point in the matrix s(ii+1)
                Trajs(jj).P(end+1) = index(index2);
                %we inform in s.isTraj that the point belongs to a trajectory
                s(ii+1).isTraj(index(index2)) = -1;
                %we put the new point in the index
                frame(ii+1).indexTraj = [frame(ii+1).indexTraj, jj];   
                
            %the choosen point belongs to the beginning of a traj 
            elseif s(ii+1).isTraj(index(index2)) >= 1
                %disp('a')
                idxTraj = s(ii+1).isTraj(index(index2));
                
                %delete in frame.indexTraj the index corresponding to 
                %the end of the old traj
                frameEnd = Trajs(idxTraj).T(end);
                idxEnd = find(frame(frameEnd).indexTraj == idxTraj);
                frame(frameEnd).indexTraj(idxEnd) = [];
                %put in frame.indexTraj the index of the current traj
                frame(frameEnd).indexTraj = [frame(frameEnd).indexTraj,jj];
                
                for jjj=1:numel(fn)
                    %we link the old traj to the current one
                    Trajs(jj).(fn{jjj}) = [Trajs(jj).(fn{jjj}),...
                        Trajs(idxTraj).(fn{jjj})];
                end
                %we delete the old traj
                deleteIdxTraj = [deleteIdxTraj,idxTraj];
                %We modify the index of s(ii+1).isTraj  
                s(ii+1).isTraj(index(index2)) = -1;
            end                
        end
    end
    frame(ii).indexTraj(deleteTraj) = [];
    %we put the index of deleteIdxTraj in descending order
    deleteIdxTraj = sort(deleteIdxTraj,'descend');
    %for all the indices of trajectories we delete
    for delIdx=deleteIdxTraj
        Trajs(delIdx) = [];
        for kkk=1:nImages
            %the index of s.isTraj >=delIdx are decreased by one
            s(kkk).isTraj(s(kkk).isTraj > delIdx) =...
            s(kkk).isTraj(s(kkk).isTraj > delIdx) - 1;    
            %the index of frame.indexTraj > delIdx are decreased by 1
            frame(kkk).indexTraj(frame(kkk).indexTraj > delIdx) =...
            frame(kkk).indexTraj(frame(kkk).indexTraj > delIdx) - 1;
        end 
    end
    
    
    %We look for the points that hasn't been linked to a trajectory,
    %and if they belong to criteria, we put them in a new trajectory
    logicalNewTraj = s(ii+1).isTraj == 0;
    logicalArea = s(ii+1).Area > 700 & s(ii+1).Area < 1300;
    
    newTraj = find(logicalNewTraj & logicalArea);
    
    for jj=newTraj
        n = length(Trajs);
        for jjj=1:numel(fn)-2
            %we add the new point in Trajs for each field of 's'
            Trajs(n+1).(fn{jjj})(1) = s(ii+1).(fn{jjj})(jj);
        end
        Trajs(n+1).T(1) = ii+1;
        Trajs(n+1).P(1) = jj;
        %the first point of the new traj takes in s.isTraj his index
        %in the struct Trajs
        s(ii+1).isTraj(jj) = n+1;
        %we add the new trajectory to the index of trajectories
        frame(ii+1).indexTraj = [frame(ii+1).indexTraj,n+1];
    end
end
end

clear ii jj jjj kkk logical1 logical2 logical3 deleteIdxTraj deleteTraj...
    delIdx dist n frameEnd idx idxEnd idxTraj index index2 logicalArea...
    logicalNewTraj R
save(fullfile('.',nameFolder,'position2.mat'),...
    's','d','Trajs','frame')

%% Loop 2
load(fullfile('.',nameFolder,'position2.mat'))

dist = 20;
for ii=1:nImages-1
    idxArea2 = find(s(ii+1).Area >  1600 & s(ii+1).Area < 2600 & s(ii+1).isTraj==0);  
    for idx=idxArea2
        %indexPos will be the index in s(ii) of the particles that end
        %a trajectory. 
        %k-th element of indexPos correspond to the k-th element of
        %frame(ii).indexTraj
        indexPos = [];
        for idxTraj=frame(ii).indexTraj
            %Verification
            if Trajs(idxTraj).T(end) == ii
                %disp('frame.indexTraj is coherent')
            else
                disp('frame.indexTraj is not coherent')
            end
            %we add the index in s(ii) of the particles which end a traj
            indexPos = [indexPos,Trajs(idxTraj).P(end)];
        end
%         disp(indexPos)
%         disp(frame(ii).indexTraj)
        %index of the particles (which end a trajectory)
        %in the circle of radius r=dist  
        index = find((s(ii+1).X(idx) - s(ii).X(indexPos)).^2 +...
                     (s(ii+1).Y(idx) - s(ii).Y(indexPos)).^2 ... 
            < dist^2);
%         disp(ii)
%         (s(ii+1).X(idx) - s(ii).X(indexPos)).^2 +...
%         (s(ii+1).Y(idx) - s(ii).Y(indexPos)).^2
        if length(index)==2
%             if s(ii).X(indexPos(index(1))) == Trajs(frame(ii).indexTraj(index(1))).X(end)...
%                     && s(ii).X(indexPos(index(2))) == Trajs(frame(ii).indexTraj(index(2))).X(end)...
%                     && s(ii).Y(indexPos(index(1))) == Trajs(frame(ii).indexTraj(index(1))).Y(end)...
%                     && s(ii).Y(indexPos(index(2))) == Trajs(frame(ii).indexTraj(index(2))).Y(end)
%                 disp('c est coherent')
%             else
%                 disp('c est pas coherent')
%             end
            [X1,Y1,X2,Y2] = move2Robots(...
                s(ii+1).X(idx)             ,s(ii+1).Y(idx),...
                s(ii).X(indexPos(index(1))),s(ii).Y(indexPos(index(1))),...
                s(ii).X(indexPos(index(2))),s(ii).Y(indexPos(index(2))) );
            %We modify s(ii+1)     
            %change the coordinates
            s(ii+1).X(idx) = X1;
            s(ii+1).X(end+1) = X2;
            s(ii+1).Y(idx) = Y1;
            s(ii+1).Y(end+1) = Y2;
            %change the area
            s(ii+1).Area(idx) = s(ii+1).Area(idx)/2;
            s(ii+1).Area(end+1) = s(ii+1).Area(idx);
            %now they belongs to a trajectory
            s(ii+1).isTraj(idx) = -1;
            s(ii+1).isTraj(end+1) = -1;
            %for the other fields, we keep the same value than the frame before
            fd = fieldnames(s);
            for kk=4:7
                s(ii+1).(fd{kk})(idx) = s(ii).(fd{kk})(indexPos(index(1)));
                s(ii+1).(fd{kk})(end+1) = s(ii).(fd{kk})(indexPos(index(2)));
            end
            
            %We add the 2 new points to Trajs
            %all fieldnames between Area and Eccentricity
            fn = fieldnames(Trajs);
            for kk=1:7
            Trajs(frame(ii).indexTraj(index(1))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(idx);
            Trajs(frame(ii).indexTraj(index(2))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(end);
            end
            %fieldname T
            Trajs(frame(ii).indexTraj(index(1))).T(end+1) = ii+1;
            Trajs(frame(ii).indexTraj(index(2))).T(end+1) = ii+1;
            %fieldname P
            Trajs(frame(ii).indexTraj(index(1))).P(end+1) = idx;
            Trajs(frame(ii).indexTraj(index(2))).P(end+1) = length(s(ii+1).Area);
            
            %We modify frame.indexTraj
            frame(ii+1).indexTraj = [frame(ii+1).indexTraj,...
                                     frame(ii  ).indexTraj(index)];
            frame(ii).indexTraj(index) = [];
            
%             disp('a')
%             fullname = fullfile(folder, D(ii).name); %full name of the image
%             bw = imread(fullname);
%             figure(1);
%             imshow(bw)
%             hold on
% %             h1 = plot([s(ii).X(indexPos(index(1))) s(ii).X(indexPos(index(2)))],...
% %                  [s(ii).Y(indexPos(index(1))) s(ii).Y(indexPos(index(2)))],...
% %                  '.','MarkerSize',10,'Color','b');
%             h2 = plot([s(ii+1).X(idx),s(ii+1).X(end)],...
%                 [s(ii+1).Y(idx),s(ii+1).Y(end)],'.','MarkerSize',10,'Color','r');
%             title(ii)
%             drawnow
%             pause(1)
            %close all
            %delete(h1);delete(h2);
%             hold off
        end            
    end
end

%% Plot
%tQueue of the trajectory plot
tQueue = 0/25;
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
for kk=14:16
    fullname = fullfile(folder, D(kk).name); %full name of the image
    bw = imread(fullname);
    figure(1);
    imshow(bw)
    hold on
    plot(X(max(1,kk-tQueue*25):kk,:), Y(max(1,kk-tQueue*25):kk,:),'.', 'MarkerSize',10);
    title(sprintf('frame=%05.0f', kk))
    pause(0.5)
    hold off
end

%% Test
disp(length(Trajs))
count = 0;
for ii = 1:length(frame)
    count = count + length(frame(ii).indexTraj);
end

% Area = zeros(150,15);
% for ii=1:150
%     for jj=1:length(s(ii).Area)
%         Area(ii,jj) = s(ii).Area(jj);
%     end
% end

% for ii=2:2
%     for jj=1:nBugs
%         if s(ii).Area(jj) > 700 && s(ii).Area(jj) < 1300
% %           pdist( [s(ii  ).X(jj),s(ii  ).Y(jj);...
% %                     s(ii-1).X(: ),s(ii-1).Y(: )] ) 
%             index = find(min(sqrt(...
%             (s(ii).X(jj)-s(ii-1).X).^2 + (s(ii).Y(jj)-s(ii-1).Y).^2 ... 
%                                   ) < dist));
%             %indexTraj = find(Trajs(:).P(end) == jj)
%         end
%      end
%             
% end


%% Save the positions of the particles

% saveFolder = fullfile('.\HeuristicPositions_2',nameFolder);
% 
% %create a folder in which the heuristic positions will be saved
% mkdir(saveFolder);
% 
% %save the heuristic positions in the right folder
% save('position.mat', 'positionX', 'positionY','Areas','Orientations',...
%     'MinorAxisLengths','MajorAxisLengths')  
% movefile('position.mat', saveFolder);

%% Show the result 
% fullname = fullfile(folder, D(image(jj)).name); %full name of the image
% bw = imread(fullname);
% imshow(bw)
% hold on
% plot(xcoords, ycoords, 'r.', 'MarkerSize',20);
% hold off

%kkk=711;
% for kkk=3212:3212%nImages%transpose(find(positionX(:,17) ~= 0))
%     fullname = fullfile(folder, D(kkk).name); %full name of the image
%     bw = imread(fullname);
%     imshow(bw)
%     hold on
%     
% %     hlen = MajorAxisLengths(kkk,:)/2;
% %     cosOrient = cosd(Orientations(kkk,:));
% %     sinOrient = sind(Orientations(kkk,:));
% %     xcoords = positionX(kkk,:)+ hlen .* [cosOrient ;-cosOrient];
% %     ycoords = positionY(kkk,:) + hlen .* [-sinOrient ;sinOrient];
% %     pl = line(xcoords, ycoords,'Color','blue','LineStyle','-','LineWidth',4);
%     
%     plot(positionX(max(1,kkk-25*2*tQueue:kkk),:), positionY(max(1,kkk-25*2*tQueue:kkk),:), 'r.', 'MarkerSize',10); 
%     
%     %F(kkk) = getframe;
%     drawnow
%     %pause(0.5)
%     hold off
% 
% end


%% Test Plot (failed)
% %Try to plot faster
% x=[];
% y=[];
% for ii=1:nImages
%     for jj=1:length(Trajs)
%         if ii==1 && Trajs(jj).T(1) == 1
%             if isempty(x)
%                 l = 1;
%             else
%                 l = length(x(1,:));
%             end
%             for kk=1:length(Trajs(jj).T)
%                 x(kk,l+1) = Trajs(jj).X(kk);
%                 y(kk,l+1) = Trajs(jj).Y(kk);
%             end
%         elseif ii ~= 1 && Trajs(jj).T(1) == ii
%             follow = find(x(ii,:)==0 & x(ii-1,:)~=0);
%             if isempty(follow) == 0
%                 for kk=1:length(Trajs(jj).T)
%                     x(ii+kk-1,min(follow)) = Trajs(jj).X(kk);
%                     y(ii+kk-1,min(follow)) = Trajs(jj).Y(kk);
%                 end
%             else
%                 l = length(x(1,:));
%                 for kk=1:length(Trajs(jj).T)
%                     x(ii+kk-1,l+1) = Trajs(jj).X(kk);
%                     y(ii+kk-1,l+1) = Trajs(jj).Y(kk);
%                 end
%             end
%         end
%     end
% end
%         
%% Test to plot faster (work very little)
% l = length(X(1,:));
% 
% for jj=1:l
%     a = 0;
%     ii = find(X(:,jj)~=0,1,'last');
%     while isempty(ii)==0 && a<=l
%         for kk=jj:l
%             if X(min(nImages,ii+1),kk) ~= 0 && X(ii,kk) == 0
%                 X(X(:,kk)~=0,jj) = X(X(:,kk)~=0,kk);
%                 Y(Y(:,kk)~=0,jj) = Y(Y(:,kk)~=0,kk);
%                 X(X(:,kk)~=0,kk) = 0;
%                 Y(Y(:,kk)~=0,kk) = 0;
%                 break
%             else 
%                 a = a + 1;
%             end
%         end
%     end
% end
% X(:,all(X==0)) = [];
% Y(:,all(Y==0)) = [];