function checkCoherence(s,d,frame,Trajs)

nImages = d.nImages;

for ii=1:nImages
    for jj=frame(ii).indexTraj
        if Trajs(jj).T(end) == ii
            %disp('cest coherent')
        else
            disp('cest pas coherent')
            Trajs(jj).T(end)
            disp(ii)
            disp(jj)
        end
    end
    for jj=1:length(Trajs)
        if s(Trajs(jj).T(1)).isTraj(Trajs(jj).P(1))==jj
            %disp('cest coherent 2')
        else
            disp('cest pas coherent 2')
        end
    end
end

disp('done, if nothing else was disp, it means it is coherent')

end