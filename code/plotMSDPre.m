%% Plot MSD form precalculated data

clc
clear varibles


inputDataFolderWildcard = 'C:\Users\THOMAS\Desktop\master_thesis_2020\msd_data\1W600C*';
inputDataFolder = 'C:\Users\THOMAS\Desktop\master_thesis_2020\msd_data\';


%convert to cell array
D = dir(inputDataFolderWildcard);
B = {D.name};

%sort names
fileNames = natsortfiles(B);
nbrExp = length(D);

%variable use in plot
np_true = 0;
na_true = 0;
mp_kg = 0;

hold on
for iExp = 1:nbrExp
    filePath = fullfile(inputDataFolder, fileNames(iExp));
    filePath = filePath{1};
    [~,name,~] = fileparts(filePath);
    data = load(filePath);
    
    fr2sec = 0.0333;
    px2m_a = 7.0137e-04;
    nbrFrames = size(data.Data.MSDActive, 1);
    
    expName_str = data.Data.name;
    mp_str = extractBefore(expName_str,"W");
    mp_kg = str2num(mp_str)*0.005 + 0.002;
    
    np_str = extractBetween(expName_str,"W","C");
    np_str = np_str{1};
    np_true = str2num(np_str);
    
    na_str = extractBetween(expName_str,"C","B");
    na_str = na_str{1};
    na_true = str2num(na_str);
    
    
    %plot
    % vary nbr bugs label
    txt = ['$N_{active}$=',num2str(na_true)];
    
    % vary weight lable
    %txt = ['$m_{passive}$=',num2str(mp_gram),'kg'];
    
    % vary nbr obstacles label need fixin
    %txt = ['$N_{active}$=',num2str(na_true)];
    
    msd = data.Data.MSDActive/na_true;
    tau = linspace(1, nbrFrames, nbrFrames)/30;
    
    loglog(tau, msd,'DisplayName',txt, 'LineWidth', 1)
    
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


