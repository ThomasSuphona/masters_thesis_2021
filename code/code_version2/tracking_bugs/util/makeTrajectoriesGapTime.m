function [s,frame,Trajs] = makeTrajectoriesGapTime(s,d,frame,Trajs,dist,gapTime)

%indices of trajs to delete in Trajs at the end
deleteIdxTraj = [];
%number of images
nImages = d.nImages;

for ii=1:nImages
    %frame(ii).indexTraj
    for jj=frame(ii).indexTraj
        for kk=ii+1:min(ii+gapTime,nImages)
            %find beginnings of trajectories
            indexBegTraj = find(s(kk).isTraj >= 1);
            %if we found a beginning of a traj
            if isempty(indexBegTraj) == 0
                if Trajs(jj).T(end) == ii
                    %disp('cest coherent')
                else
                    disp('cest pas coherent')
                end
                %position in s(ii) of the last point of Trajs(jj)
                idx = Trajs(jj).P(end);
                %find the closest traj that begin at this frame
                R = sqrt( (s(ii).X(idx) - s(kk).X(indexBegTraj)).^2 + ...
                          (s(ii).Y(idx) - s(kk).Y(indexBegTraj)).^2 );
                logical = sqrt(R) < dist;       
                index = find(logical);
                [~,index2] = min(R(index));
                
                if length(index2)==1
                    %index in Trajs of the old traj that we will link to
                    %the current one
                    idxTraj = s(kk).isTraj(indexBegTraj(index(index2)));
                    %We modify Trajs
                    fn = fieldnames(Trajs);
                    for jjj=1:numel(fn)
                        Trajs(jj).(fn{jjj}) = [Trajs(jj).(fn{jjj}),...
                                               Trajs(idxTraj).(fn{jjj})];
                    end
                    %modify s.isTraj
                    s(kk).isTraj(indexBegTraj(index(index2))) = -1;

                    %We modify frame.indexTraj
                    %delete Trajs(jj) in frame(ii)
                    frame(ii).indexTraj(frame(ii).indexTraj==jj) = [];
                    %delete Trajs(idxTraj) in frame(kk)
                    frameEnd = Trajs(idxTraj).T(end);
                    frame(frameEnd).indexTraj(frame(frameEnd).indexTraj == idxTraj)=[];
                    %add Trajs(jj) in frame(kk)
                    frame(frameEnd).indexTraj = [frame(frameEnd).indexTraj,jj];

                    %we delete the old traj
                    deleteIdxTraj = [deleteIdxTraj,idxTraj];
                    %We modify the index of s(ii+1).isTraj  
                    break
                end
            end
        end
    end
end
[s,frame,Trajs] = deleteTrajs(s,d,frame,Trajs,deleteIdxTraj);


end
