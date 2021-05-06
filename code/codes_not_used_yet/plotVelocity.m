% Plot information about the velocity
close all
%The relation between pixels and mm
px2mm = 1/1.4;
%The relation between frame and second
fr2sec = 1/30;

%the directory in which we put the folders of the tracking of robots
listing = dir('./TrackingRobots3');

for ii=[5,8,9,3,4,6,7]%[3,5:7]%[4,8:12]%length(listing)%[4,11,15,7]
%The folder in which there are the tracking files
nameFolder = ['TrackingRobots3\' listing(ii).name]
%nameFolder = 'New_Conditions_2\3W1300C25B';

%load the positions
load(fullfile('.',nameFolder,'position8.mat'))

%number of images
nImages = d.nImages;


%vX and vY will be all the velocities for each robot and each frame
% vX2 = zeros(20*nImages,1);
% vY2 = zeros(20*nImages,1);
vX = [];
vY = [];
deltaTheta = [];

for jj=1:length(Trajs)
    %for kk =1:length(Trajs(jj).X)-1
    vX = [vX, Trajs(jj).X(2:end) - Trajs(jj).X(1:end-1)];
    vY = [vY, Trajs(jj).X(2:end) - Trajs(jj).X(1:end-1)];
    deltaTheta = [deltaTheta, Trajs(jj).Orientation(2:end) - Trajs(jj).Orientation(1:end-1)];
        %vX2((jj-1)*length(Trajs(jj-1)) + kk) = Trajs(jj).X(kk+1) - Trajs(jj).X(kk);
        %vY2((jj-1)*length(Trajs(jj-1)) + kk) = Trajs(jj).X(kk+1) - Trajs(jj).X(kk);
    %end
end 
%%
vX(vX==0) = [];
%vX2(vX2==0) = [];
vY(vY==0) = [];
%vY2(vY2==0) = [];
deltaTheta(deltaTheta==0) = [];

%% In the case we change the number of cylinders
%find the number of Bugs
nBugs = str2double( regexp( listing(ii).name, ...
    '(?<=C)\d+(?=B)', 'match' ));
%we change vX and vY so that the number of data we plot in the histogram is
%weighted by the number of cylinders
vXb = vX;
vYb = vY;
deltaThetaB = deltaTheta;
vX = repmat(vX,1,floor(20/nBugs));
vY = repmat(vY,1,floor(20/nBugs));
deltaTheta = repmat(deltaTheta,1,floor(20/nBugs));
if nBugs==25
    vX = vXb(1:floor(length(vXb)*20/25));
    vY = vYb(1:floor(length(vYb)*20/25));
    deltaTheta = deltaThetaB(1:floor(length(deltaThetaB)*20/25));
elseif nBugs == 15
    vX = [vX vXb( 1:floor(length(vXb)/mod(20,nBugs)) ) ];
    vY = [vY vYb( 1:floor(length(vYb)/mod(20,nBugs)) )];    
    deltaTheta = [deltaTheta ...
    deltaThetaB( floor(length(deltaThetaB)/mod(20,nBugs)) )]; 
end
    %% Plot
figure(1);
h1 = histogram(sqrt(vX.^2+vY.^2)*px2mm/10/fr2sec,1e3,'EdgeAlpha',0.3,...
    'FaceColor','default','EdgeColor','none');
alpha(0.3) 
title('Histogram of the velocity of robots')
ylabel('N')
xlabel('v (cm/s)')
%legend('100 Cylinders','500 Cylinders','900 Cylinders','1300 Cylinders')
% legend('m_{cylinders} = 2g','m_{cylinders} = 6g','m_{cylinders} = 10g',...
%        'm_{cylinders} = 14g')
legend('1 Robot','2 Robot','5 Robot','10 Robot','15 Robot','20 Robot','25 Robot')
axis([0 40 0 1e3])
ax = gca;
ax.FontSize = 15;
%'BinLimits',[10,500]
%pause(3)
hold on
figure(2);
h2 = histogram(deltaTheta*3.1416/180/fr2sec,5e3,'EdgeAlpha',0.3,...
    'FaceColor','default','EdgeColor','none');
title('Histogram of \partial\theta/\partial t')
axis([-7 7 0 1e3])
alpha(0.3) 
ylabel('N')
xlabel('\partial\theta/\partial t (rad/s)')
%legend('100 Cylinders','500 Cylinders','900 Cylinders','1300 Cylinders')
% legend('m_{cylinders} = 2g','m_{cylinders} = 6g','m_{cylinders} = 10g',...
%        'm_{cylinders} = 14g')
legend('1 Robot','2 Robot','5 Robot','10 Robot','15 Robot','20 Robot','25 Robot')
ax = gca;
ax.FontSize = 15;
hold on

% savefig(h1,'figure1.fig')
% savefig(h2,'figure2.fig')
end
hold off