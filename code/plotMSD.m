%% plotMSD
% Try to calculate msd and plot it 
% Using the converted data

clc
clear all


inputDataFolderWildcard = 'C:\Users\THOMAS\Desktop\master_thesis_2020\main_data\1W600C*';
inputDataFolder = 'C:\Users\THOMAS\Desktop\master_thesis_2020\main_data\';


%convert to cell array
D = dir(inputDataFolderWildcard);
B = {D.name};

%sort names
fileNames = natsortfiles(B);
nbrExp = length(D);
np_true = 0;
na_true = 0;
mp_kg = 0;

tolFrames = 0.5;

hold on
for iExp = 1:nbrExp
    filePath = fullfile(inputDataFolder, fileNames(iExp));
    filePath = filePath{1};
    [~,name,~] = fileparts(filePath);
    data = load(filePath);
    
    nbrFrames = data.Data.nFrames;     % nbr frames from video
    fr2sec = 1/data.Data.FrameRate_Hz;
    
    np_true = data.Data.NP_true;
    np_tracked = data.Data.NP_tracked;
    np_use = length(data.Data.PIndices);
    pIndices = data.Data.PIndices;
    px2m_p = data.Data.pixels2meterP;
    mp_kg = data.Data.NWeights_nbr*0.005 + 0.002;
    
    na_true = data.Data.NA_true;
    na_tracked = data.Data.NA_tracked;
    na_use = length(data.Data.AIndices);
    aIndices = data.Data.AIndices;
    px2m_a = data.Data.pixels2meterA;
    
    
    
    fprintf('----------------------------------\n')
    fprintf('calculating MSD for %s\n', name);
    fprintf('nbr frames: %d\n', nbrFrames);
    fprintf('active true: %d\n', na_true);
    
    
    % Trajectories without any filters 
    % usually have more particles than the true number 
    aTrajXNoFilter = data.Data.AtrajectoriesX_px;
    aTrajYNoFilter = data.Data.AtrajectoriesY_px;
    
    
    nbrFramesA = size(aTrajXNoFilter, 2); % nbr frames from tracking
    msdSumA = zeros(nbrFramesA-1, 1);       
    msdDenom = zeros(nbrFramesA-1, 1);    % better division of average msd
    
    % After filtering out bad trajectories
    filtIndx = filterNonZero(aTrajXNoFilter, na_true, na_tracked, tolFrames);
    na_filt = length(filtIndx);
    aTrajXFilter = aTrajXNoFilter(filtIndx, :)*px2m_a;
    aTrajYFilter = aTrajYNoFilter(filtIndx, :)*px2m_a;
    
    fprintf('active filter: %d\n', na_filt);
    fprintf('----------------------------------\n')
    
    
   

    
    for iAParticle = 1:length(filtIndx)
        currATraj_x = full(aTrajXFilter(iAParticle, :));
        currATraj_y = full(aTrajYFilter(iAParticle, :));
    
        % Additional filter
        nonZeroFramesIndices = filterNonZeroFrames(currATraj_x);
        nbrNonZeroFrames = length(nonZeroFramesIndices);
        msdDenom(1:nbrNonZeroFrames-1) = msdDenom(1:nbrNonZeroFrames-1) + 1;
        
        X = currATraj_x(:, nonZeroFramesIndices);
        Y = currATraj_y(:, nonZeroFramesIndices);
        
        msdrA = msd(X, Y);
        msdSumA = msdSumA + [msdrA;zeros(length(msdSumA)-length(msdrA), 1)];
        
    end
   
    msdDenom(msdDenom<1) = 1;
    msdFinal = msdSumA./msdDenom;
    tau = linspace(1, nbrFramesA-1, nbrFramesA-1)/30;
    
    %plot
    % vary nbr bugs label
    txt = ['$N_{active}$=',num2str(na_true)];
    
    % vary weight lable
    %txt = ['$m_{passive}$=',num2str(mp_gram),'kg'];
    
    % vary nbr obstacles label need fixin
    %txt = ['$N_{active}$=',num2str(na_true)];
    
    % Filter nonzero msd
    msdIndices = filterNonZeroFrames(msdFinal');
    [M,I] = max(msdFinal);
    
    plot(tau(1:I), msdFinal(1:I),'DisplayName',txt, 'LineWidth', 1)
    
    
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
