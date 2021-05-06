function  [s,frame,Trajs]=deleteShortTraj(s,d,frame,Trajs,minLtraj)

deleteIdxTraj = [];
for jj=1:length(Trajs)
    if length(Trajs(jj).Area) <= minLtraj
        %delete the trajectory in frame.indexTraj
        frame(Trajs(jj).T(end)).indexTraj(...
        frame(Trajs(jj).T(end)).indexTraj == jj) = [];
        %the trajectory in s.isTraj is now -1
        s(Trajs(jj).T(1)).isTraj(Trajs(jj).P(1)) = -1;
        %delete the traj in Trajs and move all the indices of
        %frame.indexTraj and of s.isTraj in consequence
        deleteIdxTraj = [deleteIdxTraj,jj]; 
    end
end

[s,frame,Trajs] = deleteTrajs(s,d,frame,Trajs,deleteIdxTraj);

end