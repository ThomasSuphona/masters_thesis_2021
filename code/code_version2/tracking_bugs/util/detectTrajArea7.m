function [s,d,Trajs,frame] = detectTrajArea7(s,d,Trajs,frame,distances,minArea,maxArea)

%number of images
nImages = d.nImages;

for dist=distances
for ii=1:nImages-1
    idxArea7 = find(s(ii+1).Area >  minArea & s(ii+1).Area < maxArea & s(ii+1).isTraj==0);  
    for idx=idxArea7
        %indexPos will be the index in s(ii) of the particles that end
        %a trajectory. 
        %k-th element of indexPos correspond to the k-th element of
        %frame(ii).indexTraj
        indexPos = [];
        for idxTraj=frame(ii).indexTraj
            %we add the index in s(ii) of the particles which end a traj
            indexPos = [indexPos,Trajs(idxTraj).P(end)];
        end
        %index of the particles (which end a trajectory)
        %in the circle of radius r=dist  
        index = find((s(ii+1).X(idx) - s(ii).X(indexPos)).^2 +...
                     (s(ii+1).Y(idx) - s(ii).Y(indexPos)).^2 ... 
            < dist^2);
        % if there is nRobots particles inside the circle
        if length(index)==7
            [X1,Y1,X2,Y2,X3,Y3,X4,Y4,X5,Y5,X6,Y6,X7,Y7] = moveRobots(...
                s(ii+1).X(idx)             ,s(ii+1).Y(idx),...
                s(ii).X(indexPos(index(1))),s(ii).Y(indexPos(index(1))),...
                s(ii).X(indexPos(index(2))),s(ii).Y(indexPos(index(2))),...
                s(ii).X(indexPos(index(3))),s(ii).Y(indexPos(index(3))),...
                s(ii).X(indexPos(index(4))),s(ii).Y(indexPos(index(4))),...
                s(ii).X(indexPos(index(5))),s(ii).Y(indexPos(index(5))),...
                s(ii).X(indexPos(index(6))),s(ii).Y(indexPos(index(6))),...
                s(ii).X(indexPos(index(7))),s(ii).Y(indexPos(index(7)))...
                );
            
            %We modify s(ii+1)     
            %change the coordinates
            s(ii+1).X(idx) =   X1;
            s(ii+1).X(end+1) = X2;
            s(ii+1).X(end+1) = X3;
            s(ii+1).X(end+1) = X4;
            s(ii+1).X(end+1) = X5;
            s(ii+1).X(end+1) = X6;
            s(ii+1).X(end+1) = X7;
            
            s(ii+1).Y(idx) =   Y1;
            s(ii+1).Y(end+1) = Y2;
            s(ii+1).Y(end+1) = Y3;
            s(ii+1).Y(end+1) = Y4;
            s(ii+1).Y(end+1) = Y5;
            s(ii+1).Y(end+1) = Y6;
            s(ii+1).Y(end+1) = Y7;
            %change the area
            s(ii+1).Area(idx) = s(ii+1).Area(idx)/7;
            s(ii+1).Area(end+1) = s(ii+1).Area(idx);
            s(ii+1).Area(end+1) = s(ii+1).Area(idx);
            s(ii+1).Area(end+1) = s(ii+1).Area(idx);
            s(ii+1).Area(end+1) = s(ii+1).Area(idx);
            s(ii+1).Area(end+1) = s(ii+1).Area(idx);
            s(ii+1).Area(end+1) = s(ii+1).Area(idx);
            %now they belongs to a trajectory
            s(ii+1).isTraj(idx) = -1;
            s(ii+1).isTraj(end+1) = -1;
            s(ii+1).isTraj(end+1) = -1;
            s(ii+1).isTraj(end+1) = -1;
            s(ii+1).isTraj(end+1) = -1;
            s(ii+1).isTraj(end+1) = -1;
            s(ii+1).isTraj(end+1) = -1;
            %for the other fields, we keep the same value than the frame before
            fd = fieldnames(s);
            for kk=4:7
                s(ii+1).(fd{kk})(idx) = s(ii).(fd{kk})(indexPos(index(1)));
                s(ii+1).(fd{kk})(end+1) = s(ii).(fd{kk})(indexPos(index(2)));
                s(ii+1).(fd{kk})(end+1) = s(ii).(fd{kk})(indexPos(index(3)));
                s(ii+1).(fd{kk})(end+1) = s(ii).(fd{kk})(indexPos(index(4)));
                s(ii+1).(fd{kk})(end+1) = s(ii).(fd{kk})(indexPos(index(5)));
                s(ii+1).(fd{kk})(end+1) = s(ii).(fd{kk})(indexPos(index(6)));
                s(ii+1).(fd{kk})(end+1) = s(ii).(fd{kk})(indexPos(index(7)));
            end
            
            %We add the 4 new points to Trajs
            %all fieldnames between Area and Eccentricity
            fn = fieldnames(Trajs);
            for kk=1:7
            Trajs(frame(ii).indexTraj(index(1))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(idx);
            Trajs(frame(ii).indexTraj(index(2))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(end-5);
            Trajs(frame(ii).indexTraj(index(3))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(end-4);
            Trajs(frame(ii).indexTraj(index(4))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(end-3);
            Trajs(frame(ii).indexTraj(index(5))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(end-2);
            Trajs(frame(ii).indexTraj(index(6))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(end-1);
            Trajs(frame(ii).indexTraj(index(7))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(end);
            end
            %fieldname T
            Trajs(frame(ii).indexTraj(index(1))).T(end+1) = ii+1;
            Trajs(frame(ii).indexTraj(index(2))).T(end+1) = ii+1;
            Trajs(frame(ii).indexTraj(index(3))).T(end+1) = ii+1;
            Trajs(frame(ii).indexTraj(index(4))).T(end+1) = ii+1;
            Trajs(frame(ii).indexTraj(index(5))).T(end+1) = ii+1;
            Trajs(frame(ii).indexTraj(index(6))).T(end+1) = ii+1;
            Trajs(frame(ii).indexTraj(index(7))).T(end+1) = ii+1;
            %fieldname P
            Trajs(frame(ii).indexTraj(index(1))).P(end+1) = idx;
            Trajs(frame(ii).indexTraj(index(2))).P(end+1) = length(s(ii+1).Area)-5;
            Trajs(frame(ii).indexTraj(index(3))).P(end+1) = length(s(ii+1).Area)-4;
            Trajs(frame(ii).indexTraj(index(4))).P(end+1) = length(s(ii+1).Area)-3;
            Trajs(frame(ii).indexTraj(index(5))).P(end+1) = length(s(ii+1).Area)-2;
            Trajs(frame(ii).indexTraj(index(6))).P(end+1) = length(s(ii+1).Area)-1;
            Trajs(frame(ii).indexTraj(index(7))).P(end+1) = length(s(ii+1).Area);
  
            %We modify frame.indexTraj
            frame(ii+1).indexTraj = [frame(ii+1).indexTraj,...
                                     frame(ii  ).indexTraj(index)];
            frame(ii).indexTraj(index) = [];
        end            
    end
end
end   

end