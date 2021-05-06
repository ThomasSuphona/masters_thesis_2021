function [s,d,Trajs,frame] = detectTrajsDist(s,d,Trajs,frame,distances,minArea,maxArea)
%  . s is the struct given by the program detectBugs_3
%  . d is a struct with parameters of the experiment
%  . Trajs is the struct of all constructed trajectories
%  . frame is a struct which gives for each frame the indices of 
% trajectories that end there
%  . distances is a matrix with the maximum distances for which the 
% program will try link 2 positions into a trajectory. To avoid wrong
% trajectories to appear, it is better to put several increasing distances, 
% for example [12 16 20]. The program will turn for every distances, one by one.

%number of images
nImages = d.nImages;
%fieldnames of Trajs
fn = fieldnames(Trajs);
%index in Trajs of trajectories to delete at the end 
deleteIdxTraj = [];

for dist=distances
    dist
for ii=1:nImages-1 %for each frame
%     if floor(ii/500)==ii/500 
%         disp(ii) 
%     end
    %trajs to delete in the frame.indexTraj
    deleteTraj = [];
%     %trajs to delete in the struct Trajs
%     deleteIdxTraj = [];
    %For all trajectories in construction
    for jj=frame(ii).indexTraj
        %The index in the struct array s(ii)
        idx=Trajs(jj).P(end);
        %Find the closest point inside the circle of diameter d=dist,
        %which doesn't belong to a trajectory,
        %and which has an area between minArea and maxArea
        R = (s(ii).X(idx)-s(ii+1).X).^2 + (s(ii).Y(idx)-s(ii+1).Y).^2;
        logical1 = sqrt(R) < dist; 
        %points that belong to no trajectory or to the first point of a traj
        logical2 = s(ii+1).isTraj >= 0; 
        logical3 = s(ii+1).Area > minArea & s(ii+1).Area < maxArea;
            
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
%     %we delete the trajectories in Trajs and modify as a consequence,
%     %s.isTraj and frame.indexTraj
%     [s,frame,Trajs] = deleteTrajs(s,d,frame,Trajs,deleteIdxTraj);
    
    %We look for the points that hasn't been linked to a trajectory,
    %and if they belong to criteria, we put them in a new trajectory
    logicalNewTraj = s(ii+1).isTraj == 0;
    logicalArea = s(ii+1).Area > minArea & s(ii+1).Area < maxArea;
    
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

%we delete the trajectories in Trajs and modify as a consequence,
%s.isTraj and frame.indexTraj
[s,frame,Trajs] = deleteTrajs(s,d,frame,Trajs,deleteIdxTraj);
end
