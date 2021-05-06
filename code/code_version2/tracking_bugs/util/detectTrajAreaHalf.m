function [s,d,Trajs,frame] = detectTrajAreaHalf(s,d,Trajs,frame,distances,minArea,maxArea)

%number of images
nImages = d.nImages;


for dist=distances
for ii=1:nImages-1
    idxAreaHalf = find(s(ii+1).Area >  minArea & s(ii+1).Area < maxArea & s(ii+1).isTraj==0);  
    for idx=idxAreaHalf
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
        
        % if there is only 1 particles inside the circle
        if length(index)==1 
            %index in s(ii+1) of the close 'particles' within the area
            indexHalf = find( (s(ii+1).X(idx) - s(ii+1).X).^2 +...
                              (s(ii+1).Y(idx) - s(ii+1).Y).^2    < dist^2 ...
                           & s(ii+1).Area > minArea & s(ii+1).Area < maxArea ) ;
                       
            X = sum( s(ii+1).X(indexHalf).*s(ii+1).Area(indexHalf) )/ ...
                sum( s(ii+1).Area(indexHalf) );
            Y = sum( s(ii+1).Y(indexHalf).*s(ii+1).Area(indexHalf) )/ ...
                sum( s(ii+1).Area(indexHalf) );
            
            newArea = sum( s(ii+1).Area(indexHalf) );
            
            %We modify s(ii+1)     
            %change the coordinates
            s(ii+1).X(idx) = X;
            s(ii+1).Y(idx) = Y;
            %change the area
            s(ii+1).Area(idx) = newArea;
            %now they belong to a trajectory
            s(ii+1).isTraj(idx) = -1;
            %for the other fields, we keep the same value than the frame before
            fd = fieldnames(s);
            for kk=4:7
                s(ii+1).(fd{kk})(idx) = s(ii).(fd{kk})(indexPos(index(1)));
            end
            
            %We add the 1 new points to Trajs
            %all fieldnames between Area and Eccentricity
            fn = fieldnames(Trajs);
            for kk=1:7
            Trajs(frame(ii).indexTraj(index(1))).(fn{kk})(end+1) = s(ii+1).(fn{kk})(idx);
            end
            %fieldname T
            Trajs(frame(ii).indexTraj(index(1))).T(end+1) = ii+1;
            %fieldname P
            Trajs(frame(ii).indexTraj(index(1))).P(end+1) = idx;
            
            %We modify frame.indexTraj
            frame(ii+1).indexTraj = [frame(ii+1).indexTraj,...
                                     frame(ii  ).indexTraj(index)];
            frame(ii).indexTraj(index) = [];
        end            
    end
end
end

end