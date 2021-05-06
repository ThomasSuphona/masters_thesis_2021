%% Plot MSD using Quentins original data
clc
clear all


inputDataFolderWildcard = 'C:\Users\THOMAS\Desktop\master_thesis_2020\code\code_version2\bugs_data\1W600C*';
inputDataFolder = 'C:\Users\THOMAS\Desktop\master_thesis_2020\code\code_version2\bugs_data\';


%convert to cell array
D = dir(inputDataFolderWildcard);
B = {D.name};


%sort names
fileNames = natsortfiles(B);

%some global varibles
nbrExp = length(D);
np_true = 0;
na_true = 0;
mp_kg = 0;
px2m_a = 7.0137e-04;
tolFrames = 0.1;

hold on
for iExp = 4:4%nbrExp
    filePath = fullfile(inputDataFolder, fileNames(iExp), '\position8.mat');
    filePath = filePath{1};
    nameExp = fileNames(iExp);
    name = nameExp{1};
    
    %Video variables
    data = load(filePath);
    nbrFrames = data.d.nImages;
    
    % active varibles
    na_true = data.d.nBugs;
    na_tracked = size(data.Trajs, 2);
    msdSumA = zeros(nbrFrames-1, 1);
    
    % passive varibles
    np_true = data.d.nCylinders;
    mp_kg = data.d.nWeigths*0.005 + 0.002; % kg
    
    
    fprintf('----------------------------------\n')
    fprintf('calculating MSD for %s\n', name);
    fprintf('nbr frames: %d\n', nbrFrames);
    fprintf('active true: %d\n', na_true);
    
    for iAParticle = 1:na_tracked
       currATraj_x = data.Trajs(iAParticle).X;
       currATraj_y = data.Trajs(iAParticle).Y;
       
       msdrA = msd(currATraj_x, currATraj_y);
       msdSumA = msdSumA + msdrA;
       
    end
    
    msdFinal = msdSumA/na_tracked;
    tau = linspace(1, length(msdFinal), length(msdFinal))/30;
    
    
    %plot
    % vary nbr bugs label
    txt = ['$N_{active}$=',num2str(na_true)];
    
    % vary weight lable
    %txt = ['$m_{passive}$=',num2str(mp_gram),'kg'];
    
    % vary nbr obstacles label need fixin
    %txt = ['$N_{active}$=',num2str(na_true)];
    
    plot(tau, msdFinal,'DisplayName',txt, 'LineWidth', 1)
end


hold off
legend('show', 'Location', 'southeast', 'Interpreter','latex', 'FontSize',14)
xlabel('$\tau[s]$','Interpreter','latex', 'FontSize',14)
ylabel('MSD($\tau$)[$m^2$]','Interpreter','latex', 'FontSize',14)

% vary nbr bugs title
title(['$N_{passive}=$', num2str(np_true),' $m_{passive}$=', num2str(mp_kg), 'kg'],'Interpreter','latex','FontSize',14);

% vary weight title
%title(['$N_{passive}=$', num2str(np_true),' $N_{active}=$', num2str(na_true)],'Interpreter','latex','FontSize',14);

% vary nbr obstacles title need fixin
%title('$N_{passive}=100$, $m_{passive}=17\cdot10^{-3}$kg','Interpreter','latex','FontSize',14)
