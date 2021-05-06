function compareStruct(d,s,s2,frame,frame2,Trajs,Trajs2)

nImages = d.nImages;

for ii=1:nImages
    for jj=1:length(s2(ii).Area)
        fn = fieldnames(s);
        for kk=1:length(fn)
            if s(ii).(fn{kk})(jj) ~= s2(ii).(fn{kk})(jj)
                disp('stg wrong s')
            end   
        end
    end
    for jj=1:length(frame(ii).indexTraj)
        if frame(ii).indexTraj(jj) ~= frame2(ii).indexTraj(jj)
            disp('stg wrong indexTraj')          
        end            
    end
end
for jj=1:length(Trajs)
    fd = fieldnames(Trajs);
    for kk=1:length(fd)
        if Trajs(jj).(fd{kk}) ~= Trajs2(jj).(fd{kk})
            disp('stg wrong Traj')          
        end  
    end
end

disp('done, if nothing else was disp, everything was equal')

end
