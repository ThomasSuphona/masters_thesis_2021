%% main Plot msd from all the folders that has data
% Version 1

clc
clf
clear all

px2mm = 1/1.4;
fr2sec = 1/30;

mainPath = 'F:\thesis_2019\Version2\Code/Positions/New_Conditions_2';
D = dir(mainPath);

for iDir = 3:length(D)
    subdirPath = 'F:\thesis_2019\Version2\Code/Positions/New_Conditions_2';
    folder = sprintf('%s\\%s', subdirPath, D(iDir).name);

    if length(dir(folder)) > 2
        d = dir(folder);
        filePath = sprintf('%s\\%s', folder, d(3).name);
    end
    
    
    struct = load(filePath);
    posX = struct.X*px2mm; % [mm]
    posY = struct.Y*px2mm; % [mm]
    
    nbrFrames = size(posX, 1);
    tau = linspace(1, nbrFrames-1, nbrFrames-1);
    msdr = mean(msdn(posX, posY), 2);
    hold on
    loglog(tau*fr2sec, msdr, 'DisplayName', D(iDir).name)
 
end

hold off
set(gca, 'XScale', 'log', 'YScale', 'log');
legend('Location','southeast')
legend show
title('Mean square displacement')
xlabel('\tau [s]')
ylabel('msd(\tau) [mm^2]')


%% Calculate diffusion coefficient
% more to do here

clc
clf
clear all

nbrTaus = 30;
tau = linspace(1, nbrTaus);

mainPath = 'F:\thesis_2019\Version2\Code/Positions/New_Conditions_2';
D = dir(mainPath);
count = 0;

for iDir = 3:length(D)
    subdirPath = 'F:\thesis_2019\Version2\Code/Positions/New_Conditions_2';
    folder = sprintf('%s\\%s', subdirPath, D(iDir).name);

    if length(dir(folder)) > 2
        d = dir(folder);
        filePath = sprintf('%s\\%s', folder, d(3).name);
        count = count + 1;
    end
    
end

alphaMatrix = zeros(count, 3);

for iDir = 4:4%length(D)
    subdirPath = 'F:\thesis_2019\Version2\Code/Positions/New_Conditions_2';
    folder = sprintf('%s\\%s', subdirPath, D(iDir).name);

    if length(dir(folder)) > 2
        d = dir(folder);
        filePath = sprintf('%s\\%s', folder, d(3).name);
    end
    
    
    struct = load(filePath);
    posX = struct.X;
    posY = struct.Y;

    disp(filePath)
    msdout = msd(posX, posY, 1, tau);
    
    p = polyfit(log(tau), log(msdout)', 1);
    
    x = linspace(0, log(tau(end)));
    y = p(1)*x + p(2);
    size(posX)
    hold on
    plot(x, y, 'b')
    plot(log(tau), log(msdout), '--k')
    hold off
    p(1)
    
    %num = str2double( regexp( str, '(?<=_)\d+(?=_)', 'match' )) ;
    %disp(filePath)
    %num = str2double( regexp( D(iDir).name, '\d+(?=W)', 'match' ))
end

%% Plot msd for cylinders
% 1 file

clc
clear all

px2mm = 1/1.4;
fr2sec = 1/30;

OTS;

directory = './Cylinders';
d = dir(directory);

for iExperiment = 3:length(d)
    
    
    nbrTrueWeights = str2double( regexp( d(iExperiment).name, ...
        '\d+(?=W)', 'match' ));
    
    nbrTrueCylinders = str2double( regexp( d(iExperiment).name, ...
        '(?<=W)\d+(?=C)', 'match' ));
    
    nbrTrueBugs = str2double( regexp( d(iExperiment).name, ...
        '(?<=C)\d+(?=B)', 'match' ));
    
    filePath = fullfile('./Cylinders', d(iExperiment).name);
    traj = load(filePath);
    
    framesToTrack = traj.dvm.FramesToTrack;
    trajs = traj.dvm.Trajectories;
   
    msdsum = zeros(framesToTrack-1, 1);
    count = 0;
    
    nbrCylinders = length(trajs);
    
    for iCylinder = 1:nbrCylinders
        
        if (length(trajs(iCylinder).X) == framesToTrack)
            
            posX = trajs(iCylinder).X;
            posY = trajs(iCylinder).Y;
    
            msdsum = msdsum + msd(posX*px2mm, posY*px2mm);
            count = count + 1;
            
        end
        
    end
    
    hold on
    
    tau = linspace(1, framesToTrack-1, framesToTrack-1);
    loglog(tau.*fr2sec, msdsum./count, 'DisplayName', ...
        sprintf('%dW%dC%dB', nbrTrueWeights, nbrTrueCylinders, ...
        nbrTrueBugs));
    
    fprintf('%d/%d cylinders processed\n', count, nbrTrueCylinders);
    
end

hold off
set(gca, 'XScale', 'log', 'YScale', 'log');
legend('Location','southeast')
legend show
title('Mean square displacement')
xlabel('\tau [s]')
ylabel('msd(\tau) [mm^2]')