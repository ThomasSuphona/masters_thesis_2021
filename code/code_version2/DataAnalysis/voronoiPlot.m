% Voronoi plot and calculation

close all

%The relation between pixels and mm
px2mm = 1/1.4;
%The relation between frame and second
fr2sec = 1/30;

%the directory in which we put the folders of the tracking of robots
listing = dir('./TrackingRobots');

ii=3;
%The folder in which there are the tracking files
nameFolder = ['TrackingRobots\' listing(ii).name]
%nameFolder = 'New_Conditions_2\3W1300C25B';

%load the positions
load(fullfile('.',nameFolder,'position2.mat'))

%number of images
nImages = d.nImages;

X = zeros(nImages,length(Trajs));
Y = zeros(nImages,length(Trajs));
for jj=1:length(Trajs)
    for ii=1:length(Trajs(jj).Area)
        %For X and Y,the row is the time and the column is the trajectory
        X(Trajs(jj).T(ii),jj) = Trajs(jj).X(ii);
        Y(Trajs(jj).T(ii),jj) = Trajs(jj).Y(ii);
    end
end

%number of images
nImages = d.nImages;
% for ii=1:nImages%length(X(:,1))
%     figure(1)
%     voronoi(X(ii,X(ii,:)~=0),Y(ii,Y(ii,:)~=0))
%     xlim([0 1266])
%     ylim([0 1075])
%     drawnow
% end

for ii=1:1%length(X(:,1))
    [v,c] = voronoin([transpose(X(ii,X(ii,:)~=0))*px2mm*10,...
                      transpose(Y(ii,Y(ii,:)~=0))*px2mm*10])
end
% [vx, vy] = voronoi(X(1,:),Y(1,:));
% polyarea(vx,vy)

%x = rand(10,1) ;  y = rand(10,1) ;
%[v,c] = voronoin([x y]) 
% figure
% hold on
% voronoi(x,y)
A = zeros(length(c),1) ;
for i = 1:length(c)
    v1 = v(c{i},1) ; 
    v2 = v(c{i},2) ;
    patch(v1,v2,rand(1,3))
    A(i) = polyarea(v1,v2) ;
end


