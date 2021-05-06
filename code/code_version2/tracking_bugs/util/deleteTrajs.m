function [s,frame,Trajs] = deleteTrajs(s,d,frame,Trajs,deleteIdxTraj)
%deleteIdxTraj is the matrix of indices of all trajectories we want to
%delete in Trajs

%nImages
nImages = d.nImages;

%we put the index of deleteIdxTraj in descending order
deleteIdxTraj = sort(deleteIdxTraj,'descend');
%for all the indices of trajectories we delete
for delIdx=deleteIdxTraj
    Trajs(delIdx) = [];
    for kkk=1:nImages
        %the index of s.isTraj >=delIdx are decreased by one
        s(kkk).isTraj(s(kkk).isTraj > delIdx) =...
        s(kkk).isTraj(s(kkk).isTraj > delIdx) - 1;    
        %the index of frame.indexTraj >= delIdx are decreased by 1
        frame(kkk).indexTraj(frame(kkk).indexTraj > delIdx) =...
        frame(kkk).indexTraj(frame(kkk).indexTraj > delIdx) - 1;
    end 
end

end