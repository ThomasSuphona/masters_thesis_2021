function [s,Trajs,frame] = initTrajs(s,d,minArea,maxArea)

%nImages
nImages = d.nImages;

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
    if s(1).Area(jj) > minArea && s(1).Area(jj) < maxArea 
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
        s(1).isTraj(jj) = n;
        %We add the index of the robot in s(1) into indexTraj
        frame(1).indexTraj = [frame(1).indexTraj,n];
    end    
end

%remove the first element of Trajs, wich contains nothing
Trajs(1) = [];

end
