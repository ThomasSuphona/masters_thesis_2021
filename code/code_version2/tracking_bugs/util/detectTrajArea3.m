function [s,d,Trajs,frame] = detectTrajArea3(s,d,Trajs,frame,distances,minArea,maxArea)

%number of images
nImages = d.nImages;

for dist=distances
for ii=1:nImages-1
    idxArea3 = find(s(ii+1).Area >  minArea & s(ii+1).Area < maxArea & s(ii+1).isTraj==0);  
    for idx=idxArea3
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
        % if there is 3 particles inside the circle
        if length(index)==3
            [X1,Y1,X2,Y2,X3,Y3] = move3Robots(...
                s(ii+1).X(idx)             ,s(ii+1).Y(idx),...
                s(ii).X(indexPos(index(1))),s(ii).Y(indexPos(index(1))),...
                s(ii).X(indexPos(index(2))),s(ii).Y(indexPos(index(2))),...
                s(ii).X(indexPos(index(3))),s(ii).Y(indexPos(index(3))) );
            
            %We modify s(ii+1)     
            %change the coordinates
            s(ii+1).X(idx) =   X1;
            s(ii+1).X(end+1) = X2;
            s(ii+1).X(end+1) = X3;
            s(ii+1).Y(idx) =   Y1;
            s(ii+1).Y(end+1) = Y2;
            s(ii+1).Y(end+1) = Y3;
            %change the area
            s(ii+1).Area(idx) = s(ii+1).Area(idx)/3;
            s(ii+1).Area(end+1) = s(ii+1).Area(idx);
            s(ii+1).Area(end+1) = s(ii+1).Area(idx);
            %now they belongs to a trajectory
            s(ii+1).isTraj(idx) = -1;
            s(ii+1).isTraj(end+1) = -1;
            s(ii+1).isTraj(end+1) = -1;
            %for the other fields, we keep the same value than the frame before
            fd = fieldnames(s);
            for kk=4:7
                s(ii+1).(fd{kk})(idx) = s(ii).(fd{kk})(indexPos(index(1)));
                s(ii+1).(fd{kk})(end+1) = s(ii).(fd{kk})(indexPos(index(2)));
                s(ii+1).(fd{kk})(end+1) = s(ii).(fd{kk})(indexPos(index(3)));
            end
            
            %We add the 3 new points to Trajs
            %all fieldnames between Area and Eccentricity
            fn = fieldnames(Trajs);
            for kk=1:7
            Trajs(frame(ii).indexTraj(index(1))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(idx);
            Trajs(frame(ii).indexTraj(index(2))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(end-1);
            Trajs(frame(ii).indexTraj(index(3))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(end);
            end
            %fieldname T
            Trajs(frame(ii).indexTraj(index(1))).T(end+1) = ii+1;
            Trajs(frame(ii).indexTraj(index(2))).T(end+1) = ii+1;
            Trajs(frame(ii).indexTraj(index(3))).T(end+1) = ii+1;
            %fieldname P
            Trajs(frame(ii).indexTraj(index(1))).P(end+1) = idx;
            Trajs(frame(ii).indexTraj(index(2))).P(end+1) = length(s(ii+1).Area)-1;
            Trajs(frame(ii).indexTraj(index(3))).P(end+1) = length(s(ii+1).Area);
            %We modify frame.indexTraj
            frame(ii+1).indexTraj = [frame(ii+1).indexTraj,...
                                     frame(ii  ).indexTraj(index)];
            frame(ii).indexTraj(index) = [];
        end            
    end
end
end   

end